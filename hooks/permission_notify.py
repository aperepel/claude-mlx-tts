#!/usr/bin/env python3
"""
Permission TTS Notification - Alerts user when Claude needs permission and they may be away.

Triggers when:
- User hasn't interacted in >= IDLE_THRESHOLD_SECS seconds
- OR >= MIN_AUTO_APPROVED_TOOLS tools were auto-approved in sequence

Does NOT auto-approve - just speaks a notification and lets normal permission flow continue.
"""
import json
import os
import sys
from datetime import datetime

# Add scripts directory to path for importing TTS functions (must be before local imports)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from plugin_logging import setup_plugin_logging, LOG_DIR  # noqa: E402

# =============================================================================
# LOGGING SETUP
# =============================================================================

log = setup_plugin_logging()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Time threshold: notify if user hasn't interacted in this many seconds
# Override with PERMISSION_IDLE_THRESHOLD env var for testing
IDLE_THRESHOLD_SECS = int(os.environ.get("PERMISSION_IDLE_THRESHOLD", "30"))

# Tool count threshold: notify if this many tools were auto-approved in sequence
# Override with PERMISSION_MIN_TOOLS env var for testing
MIN_AUTO_APPROVED_TOOLS = int(os.environ.get("PERMISSION_MIN_TOOLS", "3"))

# Cooldown: don't notify more than once per this many seconds
# Override with PERMISSION_COOLDOWN env var for testing
NOTIFY_COOLDOWN_SECS = int(os.environ.get("PERMISSION_COOLDOWN", "60"))

# Path to track last notification time
NOTIFY_TIMESTAMP_FILE = LOG_DIR / ".last_permission_notify"

# =============================================================================
# IMPLEMENTATION
# =============================================================================


def get_hook_input():
    """Read hook input from stdin."""
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return {}


def is_real_user_message(entry: dict) -> bool:
    """Check if entry is a real human message (not a tool_result)."""
    if entry.get("type") != "user":
        return False
    message = entry.get("message")
    if not message or not isinstance(message, dict):
        return False
    content = message.get("content", "")
    if isinstance(content, list):
        if all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return False
    return True


def parse_transcript(transcript_path: str) -> list[dict]:
    """Parse JSONL transcript file into list of entries. Returns empty list on error."""
    entries = []
    try:
        with open(os.path.expanduser(transcript_path)) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (FileNotFoundError, IOError):
        pass
    return entries


def get_time_since_last_user_message(transcript_path: str) -> float:
    """Get seconds since the last real user message. Returns float('inf') if no messages."""
    entries = parse_transcript(transcript_path)
    if not entries:
        return float('inf')

    # Find last real user message
    for entry in reversed(entries):
        if is_real_user_message(entry):
            timestamp = entry.get("timestamp", "")
            if timestamp:
                try:
                    user_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    now = datetime.now(user_dt.tzinfo)
                    return (now - user_dt).total_seconds()
                except (ValueError, TypeError):
                    pass
            break

    return float('inf')


def count_auto_approved_tools_since_last_user(transcript_path: str) -> int:
    """Count tool calls since last user message (indicates automated run depth)."""
    entries = parse_transcript(transcript_path)

    # Find last user message index
    last_user_idx = -1
    for i in range(len(entries) - 1, -1, -1):
        if is_real_user_message(entries[i]):
            last_user_idx = i
            break

    if last_user_idx == -1:
        return 0

    # Count tool calls since then
    tool_count = 0
    for entry in entries[last_user_idx:]:
        if entry.get("type") != "assistant":
            continue
        content = entry.get("message", {}).get("content", [])
        if isinstance(content, list):
            tool_count += len([b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"])

    return tool_count


def is_within_cooldown() -> bool:
    """Check if we're within the notification cooldown period."""
    try:
        if os.path.exists(NOTIFY_TIMESTAMP_FILE):
            with open(NOTIFY_TIMESTAMP_FILE) as f:
                last_notify = float(f.read().strip())
            return (datetime.now().timestamp() - last_notify) < NOTIFY_COOLDOWN_SECS
    except (ValueError, IOError):
        pass
    return False


def record_notification():
    """Record that we just sent a notification."""
    try:
        with open(NOTIFY_TIMESTAMP_FILE, 'w') as f:
            f.write(str(datetime.now().timestamp()))
    except IOError:
        pass


def is_mlx_available() -> bool:
    """Check if MLX audio is installed and at least one voice exists."""
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        log.debug("MLX not available: mlx_audio not installed")
        return False

    try:
        from tts_config import discover_voices
        voices = discover_voices()
        if not voices:
            log.debug("MLX not available: no voices found in assets/")
            return False
        return True
    except ImportError:
        log.debug("MLX not available: tts_config not importable")
        return False


def speak_say(message: str):
    """Speak using macOS say command."""
    import subprocess
    import re
    clean_message = re.sub(r'\[[\w\s]+\]\s*', '', message)
    subprocess.Popen(
        ["say", "-v", "Daniel", "-r", "200", clean_message],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def speak_mlx(message: str, voice: str | None = None):
    """Speak using MLX voice cloning via HTTP server (non-blocking)."""
    try:
        from mlx_server_utils import speak_mlx_nonblocking
        log.info("MLX TTS (HTTP, non-blocking)")
        # Use non-blocking TTS so hook returns immediately
        speak_mlx_nonblocking(message, voice=voice)
    except Exception as e:
        log.warning(f"MLX TTS failed: {e}, falling back to macOS say")
        speak_say(message)


# Default permission phrase template (fallback if config unavailable)
PERMISSION_PHRASE_DEFAULT = "Claude needs permission to run {tool_name}."

# Conversational prefixes for question voicing
QUESTION_PREFIXES = ["So, ", "Now, ", "And ", ""]


def extract_question_text(tool_input: dict) -> str | None:
    """Extract the question text from AskUserQuestion tool input."""
    questions = tool_input.get("questions", [])
    if questions and isinstance(questions, list):
        first_q = questions[0]
        if isinstance(first_q, dict):
            return first_q.get("question", "")
    return None


def make_conversational(question: str) -> str:
    """Add conversational variation to a question for more natural voicing."""
    import random
    # Don't add prefix if question already starts with common words
    lower_q = question.lower()
    if any(lower_q.startswith(w) for w in ["so,", "now,", "and ", "let's", "what's"]):
        return question
    prefix = random.choice(QUESTION_PREFIXES)
    return f"{prefix}{question}"


def speak_question(question: str):
    """Voice a question using TTS with conversational variation."""
    conversational = make_conversational(question)

    if is_mlx_available():
        try:
            from tts_config import get_effective_hook_voice
            voice = get_effective_hook_voice("interview_question")
        except (ImportError, KeyError):
            voice = None
        log.info(f"Interview question [{voice}]: {conversational[:60]}...")
        speak_mlx(conversational, voice=voice)
    else:
        log.info(f"Interview question [Daniel]: {conversational[:60]}...")
        speak_say(conversational)


def speak_notification(tool_name: str):
    """Speak the permission notification using TTS."""
    # Get phrase template from config (with fallback for standalone usage)
    try:
        from tts_config import get_effective_hook_prompt
        phrase_template = get_effective_hook_prompt("permission_request")
    except ImportError:
        phrase_template = PERMISSION_PHRASE_DEFAULT

    message = phrase_template.format(tool_name=tool_name)

    if is_mlx_available():
        try:
            from tts_config import get_effective_hook_voice
            voice = get_effective_hook_voice("permission_request")
        except ImportError:
            voice = None
        log.info(f"TTS [{voice}]: {message}")
        speak_mlx(message, voice=voice)
    else:
        log.info(f"TTS [Daniel] (MLX unavailable): {message}")
        speak_say(message)


def main():
    log.info("Permission hook invoked")
    hook_input = get_hook_input()

    # Log full hook input for debugging (to understand what fields are available)
    tool_name = hook_input.get("tool_name", "unknown")
    log.info(f"Tool name from hook_input: {tool_name}")
    log.info(f"Full hook_input keys: {list(hook_input.keys())}")
    log.debug(f"Full hook_input: {json.dumps(hook_input, default=str)}")

    # Skip TTS for TTS-related commands (they'll be auto-approved by approve-tts.py)
    tool_input = hook_input.get("tool_input", {})
    command = tool_input.get("command", "")
    TTS_SCRIPTS = ["tts-init.sh", "tts-start.sh", "tts-stop.sh", "tts-status.sh", "tts-mute.sh", "run-tts.sh", "say.sh", "summary-say.sh"]
    if tool_name == "Bash" and any(script in command for script in TTS_SCRIPTS):
        log.info(f"Skipping TTS notification for auto-approved command: {command[:50]}")
        sys.exit(1)

    transcript_path = hook_input.get("transcript_path", "")

    if not transcript_path:
        log.warning("No transcript_path in hook input")
        sys.exit(1)

    # Check if TTS is muted
    try:
        from tts_mute import is_muted, get_mute_status, format_remaining_time
        if is_muted():
            status = get_mute_status()
            remaining = format_remaining_time(status.remaining_seconds)
            log.info(f"TTS muted ({remaining} remaining), skipping")
            sys.exit(1)
    except ImportError:
        pass  # tts_mute not available, continue normally

    # Voice AskUserQuestion questions directly (more reliable than skill invocation)
    # This ensures interview questions are always voiced regardless of whether
    # Claude remembers to invoke /say
    if tool_name == "AskUserQuestion":
        question_text = extract_question_text(tool_input)
        if question_text:
            log.info("AskUserQuestion detected, voicing question")
            speak_question(question_text)
        else:
            log.info("AskUserQuestion with no extractable question, skipping TTS")
        sys.exit(1)

    # Check cooldown (for non-interview permission notifications)
    if is_within_cooldown():
        log.info("Within cooldown period, skipping notification")
        sys.exit(1)

    # Check heuristics
    time_since_user = get_time_since_last_user_message(transcript_path)
    tool_count = count_auto_approved_tools_since_last_user(transcript_path)

    log.info(f"Heuristics: time_since_user={time_since_user:.1f}s, auto_approved_tools={tool_count}")

    # Don't notify if we couldn't read the transcript (inf means no valid data)
    if time_since_user == float('inf'):
        log.info("No valid transcript data, skipping notification")
        sys.exit(1)

    should_notify = (
        time_since_user >= IDLE_THRESHOLD_SECS or
        tool_count >= MIN_AUTO_APPROVED_TOOLS
    )

    # Standard permission notification for other tools
    if should_notify:
        log.info(f"Triggering notification for tool: {tool_name}")
        record_notification()
        speak_notification(tool_name)
    else:
        log.info("Heuristics not met, skipping notification")

    # Exit without output - let normal permission flow continue
    # (don't auto-approve, just notify)
    sys.exit(1)


if __name__ == "__main__":
    main()
