#!/bin/bash
# Stop the TTS server to reclaim memory
# Usage: ./tts-stop.sh
#
# This script stops the mlx_audio.server running on port 21099.
# Use this when you're done with TTS and want to reclaim GPU memory (up to 4GB depending on model).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PLUGIN_ROOT/.venv/bin/python"

# Use venv Python if available, otherwise system Python
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="python3"
fi

exec "$PYTHON" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from mlx_server_utils import stop_server, get_server_status

status = get_server_status()
if status['running']:
    if stop_server():
        print('TTS server stopped. GPU memory reclaimed.')
    else:
        print('Failed to stop TTS server.')
        sys.exit(1)
else:
    print('TTS server is not running.')
"
