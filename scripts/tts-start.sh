#!/bin/bash
# Start the TTS server to keep the model warm
# Usage: ./tts-start.sh
#
# Pre-warms the mlx_audio.server for faster TTS responses.
# Server runs on port 21099 until manually stopped with /tts-stop.

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
from mlx_server_utils import ensure_server_running, get_server_status, ServerStartError

from voice_cache import is_cache_valid
import os

voice_ref = os.environ.get('MLX_TTS_VOICE_REF', os.path.join('$PLUGIN_ROOT', 'assets', 'default_voice.wav'))
cache_exists = is_cache_valid(voice_ref)

status = get_server_status()
if status['running']:
    print(f\"TTS server already running on port {status['port']}\")
else:
    print('Starting TTS server...')
    if cache_exists:
        print('Loading cached voice embeddings...')
    else:
        print('Caching voice embeddings (one-time, ~2s)...')
    try:
        ensure_server_running()
        print(f\"TTS server ready on port {get_server_status()['port']}\")
        print('Run /tts-stop to reclaim GPU memory when done.')
    except ServerStartError as e:
        print(f'Failed to start server: {e}')
        sys.exit(1)
"
