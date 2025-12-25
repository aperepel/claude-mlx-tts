#!/bin/bash
# Check TTS server status
# Usage: ./tts-status.sh
#
# Shows whether the mlx_audio.server is running and on which port.

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
from mlx_server_utils import get_server_status

status = get_server_status()
if status['running']:
    print(f\"TTS server: RUNNING on port {status['port']}\")
    print(f\"Model: {status['model']}\")
else:
    print('TTS server: NOT RUNNING')
    print('Use /tts-start to warm up the server.')
"
