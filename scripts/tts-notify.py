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
import re
import subprocess
import os
import sys
from datetime import datetime

# =============================================================================
# LOGGING SETUP
# =============================================================================

from plugin_logging import setup_plugin_logging

log = setup_plugin_logging()

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

# Attention prefix (heads-up before content) - fallback if config unavailable
ATTENTION_PREFIX_DEFAULT = "[clear throat] Attention on deck."

# MLX Voice Cloning settings
MLX_MODEL = "mlx-community/chatterbox-turbo-fp16"
# Default voice name (without extension) - uses assets/default.safetensors
DEFAULT_VOICE = "default"


# =============================================================================
# TTS BACKENDS
# =============================================================================

# Use HTTP server by default for warm latency, with direct API as fallback
USE_HTTP_SERVER = os.environ.get("TTS_USE_HTTP", "true").lower() == "true"


def _generate_mlx_speech_direct(text: str, voice_name: str | None = None, play: bool = True):
    """Generate speech using direct MLX API with metrics logging."""
    if not text or not text.strip():
        return

    from mlx_tts_core import generate_speech, get_model

    model = get_model()
    # Uses mlx_tts_core which logs metrics (TTFT, gen_time, etc.)
    # Note: Speed not supported by Chatterbox model
    generate_speech(
        text=text,
        model=model,
        voice_name=voice_name,  # Uses active voice from config if None
        ref_text=".",
        play=play,
        stream=True,
    )


def _generate_mlx_speech_http(text: str, voice_name: str | None = None):
    """Generate speech using HTTP server (fast warm latency, non-blocking)."""
    if not text or not text.strip():
        return

    from mlx_server_utils import speak_mlx_nonblocking

    # Use non-blocking TTS so hook returns immediately
    speak_mlx_nonblocking(text, voice=voice_name)


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def is_mlx_available() -> bool:
    """Check if MLX audio is installed and at least one voice exists."""
    try:
        import mlx_audio  # noqa: F401
        from tts_config import discover_voices
        return len(discover_voices()) > 0
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
    user_text_lower = str(user_content).lower().strip()

    # Option A: Skip TTS if this turn ran a TTS script via Bash
    # Check assistant tool calls for our script names (more reliable than parsing user message)
    TTS_SCRIPTS = ["say.sh", "summary-say.sh", "tts-start.sh", "tts-stop.sh", "tts-status.sh", "tts-init.sh", "tts-mute.sh"]
    for entry in entries[last_user_idx:]:
        if entry.get("type") != "assistant":
            continue
        content = entry.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") != "Bash":
                continue
            command = block.get("input", {}).get("command", "")
            if any(script in command for script in TTS_SCRIPTS):
                log.info("Skipping TTS: turn included TTS script in Bash call")
                return False, "", 0, 0.0, False

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


def speak_mlx(message: str, hook_type: str = "stop"):
    """Speak using MLX voice cloning (HTTP server or direct API)."""
    try:
        # Get effective voice for this hook (per-hook override or default)
        from tts_config import get_effective_hook_voice
        voice_name = get_effective_hook_voice(hook_type)
        log.info(f"TTS [{voice_name}] hook={hook_type}")

        if USE_HTTP_SERVER:
            log.info("MLX TTS (HTTP)")
            _generate_mlx_speech_http(message, voice_name=voice_name)
        else:
            log.info("MLX TTS (direct)")
            _generate_mlx_speech_direct(message, voice_name=voice_name, play=True)
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

    # Option B: Prevent infinite recursion via stop_hook_active flag
    if hook_input.get("stop_hook_active", False):
        log.info("Stop hook already active, preventing recursion")
        return

    # Check if TTS is muted
    try:
        from tts_mute import is_muted, get_mute_status, format_remaining_time
        if is_muted():
            status = get_mute_status()
            remaining = format_remaining_time(status.remaining_seconds)
            log.info(f"TTS muted ({remaining} remaining), skipping")
            return
    except ImportError:
        pass  # tts_mute not available, continue normally

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

    # Get attention grabber from config (with fallback for standalone usage)
    try:
        from tts_config import get_effective_hook_prompt
        attention_grabber = get_effective_hook_prompt("stop")
    except ImportError:
        attention_grabber = ATTENTION_PREFIX_DEFAULT

    message = f"{attention_grabber} ... {summary}"
    log.info(f"Speaking: {message[:100]}...")
    speak(message)
    log.info("TTS complete")


if __name__ == "__main__":
    main()
