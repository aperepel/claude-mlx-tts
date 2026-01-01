#!/usr/bin/env python3
"""
Permission TTS Notification - Alerts user when Claude needs permission and they may be away.

Triggers when:
- User hasn't interacted in >= IDLE_THRESHOLD_SECS seconds
- OR >= MIN_AUTO_APPROVED_TOOLS tools were auto-approved in sequence

Does NOT auto-approve - just speaks a notification and lets normal permission flow continue.
"""
import json
import logging
import os
import sys
from datetime import datetime

# Add scripts directory to path for importing TTS functions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# =============================================================================
# LOGGING SETUP
# =============================================================================

LOG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "logs")
LOG_FILE = os.path.join(LOG_DIR, "permission-notify.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
    ]
)
log = logging.getLogger(__name__)

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
NOTIFY_TIMESTAMP_FILE = os.path.join(LOG_DIR, ".last_permission_notify")

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


def extract_tool_name(message: str) -> str:
    """Extract tool name from permission message."""
    # Message format: "Claude wants to use Bash" or similar
    if "Bash" in message:
        return "Bash"
    if "Write" in message:
        return "Write"
    if "Edit" in message:
        return "Edit"
    if "Read" in message:
        return "Read"
    if "WebFetch" in message:
        return "WebFetch"
    if "Task" in message:
        return "Task"
    return "a tool"


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
    """Speak using MLX voice cloning via HTTP server."""
    try:
        from mlx_server_utils import speak_mlx_http
        log.info("MLX TTS (HTTP)")
        speak_mlx_http(message, voice=voice)
    except Exception as e:
        log.warning(f"MLX TTS failed: {e}, falling back to macOS say")
        speak_say(message)


# Default permission phrase template (fallback if config unavailable)
PERMISSION_PHRASE_DEFAULT = "Claude needs permission to run {tool_name}."


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

    transcript_path = hook_input.get("transcript_path", "")
    message = hook_input.get("message", "")

    if not transcript_path:
        log.warning("No transcript_path in hook input")
        sys.exit(1)

    # Check cooldown
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

    if should_notify:
        tool_name = extract_tool_name(message)
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
