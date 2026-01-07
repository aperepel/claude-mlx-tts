#!/bin/bash
# Temporarily mute TTS notifications
# Usage: ./tts-mute.sh [duration]
#   duration: Natural language like "30 minutes", "until 3pm", "for an hour"
#             Or unmute keywords: "resume", "off", "unmute", "cancel"
#             Empty = mute indefinitely

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PLUGIN_ROOT/.venv/bin/python"

# Use venv Python if available, otherwise system Python
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="python3"
fi

# Get duration from arguments (can be empty for indefinite mute)
DURATION="$*"

exec "$PYTHON" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')

from tts_mute import set_mute, get_mute_status, format_remaining_time, clear_mute, UNMUTE_KEYWORDS

duration = '''$DURATION'''.strip()

# Handle unmute keywords specially
if duration.lower() in UNMUTE_KEYWORDS:
    clear_mute()
    print('TTS notifications resumed.')
else:
    try:
        expires_at = set_mute(duration if duration else None)
        status = get_mute_status()

        if not status.is_muted:
            # This shouldn't happen after set_mute, but handle it
            print('TTS notifications resumed.')
        elif status.expires_at is None:
            print('TTS notifications muted indefinitely.')
            print('To resume: /tts-mute resume')
        else:
            remaining = format_remaining_time(status.remaining_seconds)
            from datetime import datetime
            expires_dt = datetime.fromtimestamp(status.expires_at)
            print(f'TTS notifications muted for {remaining}.')
            print(f'Expires at: {expires_dt.strftime(\"%I:%M %p\")}')
            print('To resume early: /tts-mute resume')
    except ValueError as e:
        print(f'Error: {e}')
        sys.exit(1)
"
