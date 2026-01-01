"""
TTS Mute module.

Manages temporary muting of TTS notifications using file-based state.
Mute state is stored at ${PLUGIN_ROOT}/.config/mute_until

File format:
    - File does not exist: Not muted
    - File exists but empty: Muted indefinitely
    - File contains Unix timestamp: Muted until that time

Supports natural language duration parsing via Claude CLI (haiku model).
"""
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# Plugin-local config and logging directories
_PLUGIN_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _PLUGIN_ROOT / ".config"
MUTE_FILE = _CONFIG_DIR / "mute_until"

# Logging setup
LOG_DIR = _PLUGIN_ROOT / "logs"
LOG_FILE = LOG_DIR / "tts-mute.log"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)],
)
log = logging.getLogger(__name__)

# Keywords that trigger unmute
UNMUTE_KEYWORDS = ["resume", "off", "unmute", "cancel"]


class MuteStatus(NamedTuple):
    """Status of TTS mute state."""

    is_muted: bool
    expires_at: float | None  # Unix timestamp, None = indefinite
    remaining_seconds: float | None  # Seconds until expiry, None = indefinite


def is_muted() -> bool:
    """Check if TTS is currently muted.

    Returns:
        True if muted (file exists and not expired), False otherwise.
    """
    return get_mute_status().is_muted


def clear_mute() -> None:
    """Clear the mute state (unmute TTS).

    Removes the mute file if it exists.
    """
    try:
        MUTE_FILE.unlink()
        log.info("TTS mute cleared")
    except FileNotFoundError:
        pass  # Already cleared
    except OSError as e:
        log.warning(f"Failed to clear mute file: {e}")


def get_mute_status() -> MuteStatus:
    """Get detailed mute status.

    Returns:
        MuteStatus with:
            - is_muted: Whether TTS is currently muted
            - expires_at: Unix timestamp when mute expires (None if indefinite)
            - remaining_seconds: Seconds until expiry (None if indefinite)
    """
    try:
        content = MUTE_FILE.read_text().strip()
        if not content:
            # Empty file = indefinite mute
            return MuteStatus(is_muted=True, expires_at=None, remaining_seconds=None)

        expires_at = float(content)
        now = time.time()

        if now >= expires_at:
            # Expired, clean up and return not muted
            clear_mute()
            return MuteStatus(is_muted=False, expires_at=None, remaining_seconds=None)

        remaining = expires_at - now
        return MuteStatus(
            is_muted=True, expires_at=expires_at, remaining_seconds=remaining
        )
    except FileNotFoundError:
        return MuteStatus(is_muted=False, expires_at=None, remaining_seconds=None)
    except (ValueError, OSError):
        return MuteStatus(is_muted=False, expires_at=None, remaining_seconds=None)


def _parse_duration_nlp(phrase: str) -> float:
    """Parse a natural language duration phrase using Claude CLI.

    Uses haiku model for fast, cheap NLP parsing.

    Args:
        phrase: Natural language duration (e.g., "30 minutes", "until 3pm")

    Returns:
        Unix timestamp when mute should expire.

    Raises:
        ValueError: If parsing fails or Claude CLI returns invalid output.
    """
    now = time.time()
    now_dt = datetime.now()
    now_iso = now_dt.isoformat()
    tz_name = now_dt.astimezone().tzname() or "local"

    prompt = f"""Parse this duration and return ONLY a Unix timestamp (seconds since epoch).
Current time: {now:.0f} ({now_iso} {tz_name})
Duration phrase: "{phrase}"

If relative (e.g., "30 minutes", "for an hour"), add to current time.
If absolute (e.g., "until 3pm", "until midnight"), calculate timestamp for today (or tomorrow if time has passed).
Return ONLY the numeric timestamp, nothing else."""

    # Use clean TMPDIR to avoid Bun socket watching bug
    tmp_dir = "/tmp/claude-tts-tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = tmp_dir

    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                "haiku",
                "--settings",
                '{"hooks":{},"alwaysThinkingEnabled":false}',
                "--no-session-persistence",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "unknown error"
            raise ValueError(f"Claude CLI failed: {stderr}")

        output = result.stdout.strip()
        if not output:
            raise ValueError("Claude CLI returned empty response")

        # Parse the timestamp from output
        # Handle case where Claude might include extra text
        # Try to extract first number that looks like a Unix timestamp
        match = re.search(r"\b(\d{10,}(?:\.\d+)?)\b", output)
        if match:
            timestamp = float(match.group(1))
        else:
            # Try parsing the entire output as a number
            try:
                timestamp = float(output)
            except ValueError:
                raise ValueError(
                    f"Could not parse duration from: {output!r}. "
                    f"Try a clearer phrase like 'for 30 minutes' or 'until 5pm'."
                )

        # Validate timestamp is in the future
        if timestamp <= now:
            raise ValueError(
                f"Parsed timestamp {timestamp} is not in the future. "
                f"Try a clearer phrase like 'for 30 minutes' or 'until 5pm'."
            )

        return timestamp

    except subprocess.TimeoutExpired:
        raise ValueError("Claude CLI timed out while parsing duration")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse duration: {e}")


def set_mute(duration_phrase: str | None = None) -> float | None:
    """Set TTS mute state.

    Args:
        duration_phrase: Natural language duration (e.g., "30 minutes", "until 3pm").
            If None or empty, mutes indefinitely.
            If an unmute keyword ('resume', 'off', 'unmute', 'cancel'), clears mute.

    Returns:
        Unix timestamp when mute expires, or None if indefinite mute.

    Raises:
        ValueError: If duration parsing fails.
    """
    # Ensure config directory exists
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Handle None/empty = indefinite mute
    if not duration_phrase or not duration_phrase.strip():
        MUTE_FILE.write_text("")
        log.info("TTS muted indefinitely")
        return None

    phrase = duration_phrase.strip().lower()

    # Check for unmute keywords
    if phrase in UNMUTE_KEYWORDS:
        clear_mute()
        return None

    # Parse duration using NLP
    expires_at = _parse_duration_nlp(phrase)

    # Write timestamp to file
    MUTE_FILE.write_text(str(expires_at))

    # Log human-readable expiry
    expires_dt = datetime.fromtimestamp(expires_at)
    log.info(f"TTS muted until {expires_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    return expires_at


def format_remaining_time(seconds: float | None) -> str:
    """Format remaining seconds as human-readable string.

    Args:
        seconds: Remaining seconds, or None for indefinite.

    Returns:
        Human-readable string like "5 minutes", "2 hours", or "indefinitely".
    """
    if seconds is None:
        return "indefinitely"

    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"
