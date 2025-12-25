#!/usr/bin/env python3
"""
Claude Summary TTS - Speaks a summary when Claude finishes deep work.

Triggers when:
- Response took >= MIN_DURATION_SECS seconds
- OR >= MIN_TOOL_CALLS tool calls were made
- OR user message contained thinking keywords (think/ultrathink)

Uses macOS 'say' by default. Install MLX extra for voice cloning: uv sync --extra mlx
"""
import json
import logging
import re
import subprocess
import os
import sys
from datetime import datetime

# =============================================================================
# LOGGING SETUP
# =============================================================================

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "tts-notify.log")

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
# CONFIGURATION - Edit these to customize behavior
# =============================================================================

# Thresholds for triggering TTS
MIN_DURATION_SECS = 15      # Trigger if response took this long
MIN_TOOL_CALLS = 2          # OR if this many tool calls were made
THINKING_KEYWORDS = ["ultrathink", "think harder", "think hard", "think"]

# macOS 'say' settings (default TTS)
SAY_VOICE = "Daniel"        # Try: say -v ? to list voices
SAY_RATE = 180              # Words per minute

# Attention prefix (heads-up before content)
ATTENTION_PREFIX = "[clear throat] Attention on deck."

# MLX Voice Cloning settings
MLX_MODEL = "mlx-community/chatterbox-turbo-fp16"
MLX_SPEED = 1.6             # Speech speed multiplier (0.5-2.0)
# Voice reference: bundled in assets/, replace with your own if desired
MLX_VOICE_REF = os.path.join(os.path.dirname(__file__), "..", "assets", "default_voice.wav")

# =============================================================================
# MLX TTS CORE (inlined for hook isolation compatibility)
# =============================================================================

_cached_model = None


def _get_mlx_model():
    """Get cached MLX model, loading if necessary."""
    global _cached_model
    if _cached_model is None:
        from mlx_audio.tts.utils import load_model
        log.info(f"Loading MLX model: {MLX_MODEL}")
        _cached_model = load_model(model_path=MLX_MODEL)
        log.info("MLX model loaded")
    return _cached_model


def _generate_mlx_speech(text: str, play: bool = True):
    """Generate speech using direct MLX API."""
    if not text or not text.strip():
        return

    from mlx_audio.tts.generate import generate_audio
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = _get_mlx_model()
    generate_audio(
        text=text,
        model=model,
        ref_audio=MLX_VOICE_REF,
        ref_text=".",
        speed=MLX_SPEED,
        play=play,
        verbose=False,
    )


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def is_mlx_available() -> bool:
    """Check if MLX audio is installed and voice reference exists."""
    try:
        import mlx_audio  # noqa: F401
        return os.path.exists(MLX_VOICE_REF)
    except ImportError:
        return False


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
    content = entry.get("message", {}).get("content", "")
    # Tool results are stored as list of dicts with type "tool_result"
    if isinstance(content, list):
        if all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return False
    return True


def should_trigger_tts(transcript_path: str) -> tuple[bool, str, int, float, bool]:
    """Check if TTS should trigger. Returns (should_trigger, last_message, tool_count, duration, thinking)."""
    entries = []
    try:
        with open(os.path.expanduser(transcript_path)) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (FileNotFoundError, IOError):
        return False, "", 0, 0.0, False

    if not entries:
        return False, "", 0, 0.0, False

    # Find last REAL user message (not tool_result) - start of current turn
    last_user_idx = -1
    for i in range(len(entries) - 1, -1, -1):
        if is_real_user_message(entries[i]):
            last_user_idx = i
            break

    if last_user_idx == -1:
        return False, "", 0, 0.0, False

    # Check for thinking keywords in user message
    user_entry = entries[last_user_idx]
    user_content = user_entry.get("message", {}).get("content", "")
    if isinstance(user_content, list):
        user_content = " ".join(
            str(b.get("content", b.get("text", "")))
            for b in user_content if isinstance(b, dict)
        )
    user_text_lower = str(user_content).lower()
    thinking_triggered = any(kw in user_text_lower for kw in THINKING_KEYWORDS)

    # Get assistant entries after last user message
    assistant_entries = [e for e in entries[last_user_idx:] if e.get("type") == "assistant"]

    if not assistant_entries:
        return False, "", 0, 0.0, thinking_triggered

    # Extract last message text
    last_message = ""
    for entry in assistant_entries:
        message = entry.get("message", {})
        content = message.get("content", [])
        if isinstance(content, str):
            last_message = content
        elif isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                last_message = " ".join(texts)

    # Count tool calls
    tool_count = 0
    for entry in assistant_entries:
        message = entry.get("message", {})
        content = message.get("content", [])
        if isinstance(content, list):
            tool_count += len([b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"])

    # Calculate duration
    duration_secs = 0.0
    try:
        user_ts = entries[last_user_idx].get("timestamp", "")
        last_ts = assistant_entries[-1].get("timestamp", "")
        if user_ts and last_ts:
            user_dt = datetime.fromisoformat(user_ts.replace("Z", "+00:00"))
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            duration_secs = (last_dt - user_dt).total_seconds()
    except (ValueError, TypeError):
        pass

    # Check thresholds
    should_trigger = (
        duration_secs >= MIN_DURATION_SECS or
        tool_count >= MIN_TOOL_CALLS or
        thinking_triggered
    )

    return should_trigger, last_message[:3000], tool_count, duration_secs, thinking_triggered


def summarize(text: str) -> str:
    """Summarize text using Claude CLI."""
    prompt = f"Convert to ONE short spoken sentence (max 15 words). No intro, no quotes, just the summary:\n\n{text[:1500]}"

    # Use clean TMPDIR to avoid Bun socket watching bug
    tmp_dir = "/tmp/claude-tts-tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = tmp_dir

    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--settings", '{"hooks":{},"alwaysThinkingEnabled":false}',
                "--no-session-persistence",
                prompt
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    return "Claude has completed its task."


def speak_say(message: str):
    """Speak using macOS say command."""
    clean_message = re.sub(r'\[[\w\s]+\]\s*', '', message)
    subprocess.Popen(
        ["say", "-v", SAY_VOICE, "-r", str(SAY_RATE), clean_message],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def speak_mlx(message: str):
    """Speak using MLX voice cloning via direct API."""
    try:
        log.info(f"MLX TTS: generating speech (speed={MLX_SPEED})")
        _generate_mlx_speech(message, play=True)
        log.info("MLX TTS: complete")
    except Exception as e:
        log.warning(f"MLX TTS failed: {e}, falling back to macOS say")
        speak_say(message)


def speak(message: str):
    """Speak message using configured TTS."""
    if is_mlx_available():
        speak_mlx(message)
    else:
        speak_say(message)


def main():
    log.info("Hook invoked")
    hook_input = get_hook_input()
    transcript_path = hook_input.get("transcript_path", "")

    if not transcript_path:
        log.warning("No transcript_path in hook input")
        return

    should_trigger, last_message, tool_count, duration, thinking = should_trigger_tts(transcript_path)

    log.info(f"Threshold check: trigger={should_trigger}, duration={duration:.1f}s, tools={tool_count}, thinking={thinking}")

    if not should_trigger or not last_message:
        return

    log.info("Generating summary...")
    summary = summarize(last_message)
    message = f"{ATTENTION_PREFIX} ... {summary}"
    log.info(f"Speaking: {message[:100]}...")
    speak(message)
    log.info("TTS complete")


if __name__ == "__main__":
    main()
