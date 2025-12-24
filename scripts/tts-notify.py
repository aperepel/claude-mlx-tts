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
import subprocess
import os
import sys
from datetime import datetime

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

# MLX Voice Cloning (auto-enabled when voice reference exists)
# Model ID from HuggingFace - downloads automatically on first use (~4GB for fp16)
MLX_MODEL = "mlx-community/chatterbox-turbo-fp16"
# Voice reference: 10-20 second WAV of your voice (or any voice to clone)
MLX_VOICE_REF = "~/.config/claude-tts/voice_ref.wav"
MLX_SPEED = 1.6

# =============================================================================
# IMPLEMENTATION
# =============================================================================

def is_mlx_available() -> bool:
    """Check if MLX audio is installed and voice reference exists."""
    try:
        import mlx_audio  # noqa: F401
        voice_ref = os.path.expanduser(MLX_VOICE_REF)
        return os.path.exists(voice_ref)
    except ImportError:
        return False


def get_hook_input():
    """Read hook input from stdin."""
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return {}


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

    # Find last user message (start of current turn)
    last_user_idx = -1
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("type") == "user":
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
    subprocess.Popen(
        ["say", "-v", SAY_VOICE, "-r", str(SAY_RATE), message],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def speak_mlx(message: str):
    """Speak using MLX voice cloning."""
    output_dir = "/tmp/claude-tts"
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    try:
        subprocess.run(
            [
                sys.executable, "-m", "mlx_audio.tts.generate",
                "--model", MLX_MODEL,
                "--text", message,
                "--ref_audio", os.path.expanduser(MLX_VOICE_REF),
                "--ref_text", ".",
                "--speed", str(MLX_SPEED),
                "--file_prefix", os.path.join(output_dir, "notification"),
                "--play"
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )
    except Exception:
        speak_say(message)


def speak(message: str):
    """Speak message using configured TTS."""
    if is_mlx_available():
        speak_mlx(message)
    else:
        speak_say(message)


def main():
    hook_input = get_hook_input()
    transcript_path = hook_input.get("transcript_path", "")

    if not transcript_path:
        return

    should_trigger, last_message, tool_count, duration, thinking = should_trigger_tts(transcript_path)

    if not should_trigger or not last_message:
        return

    summary = summarize(last_message)
    speak(summary)


if __name__ == "__main__":
    main()
