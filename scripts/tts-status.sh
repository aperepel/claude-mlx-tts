#!/bin/bash
# Display TTS configuration status
# Usage: ./tts-status.sh
#
# Shows comprehensive TTS configuration including active voice, available voices,
# hook overrides, server status, and audio processing settings.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PLUGIN_ROOT/.venv/bin/python"

# Use venv Python if available, otherwise system Python
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="python3"
fi

exec "$PYTHON" "$SCRIPT_DIR/tts_config.py" full-status
