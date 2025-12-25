#!/bin/bash
# Speak text directly using TTS
# Usage: ./say.sh <text to speak>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PLUGIN_ROOT/.venv/bin/python"

# Use venv Python if available, otherwise system Python
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="python3"
fi

# Get text from arguments
TEXT="$*"

if [ -z "$TEXT" ]; then
    echo "Usage: say <text to speak>"
    exit 1
fi

exec "$PYTHON" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')

# Import speak function inline to avoid hyphen import issue
import subprocess
import os
import re

SAY_VOICE = 'Daniel'
SAY_RATE = 180
MLX_SPEED = 1.6
MLX_VOICE_REF = os.path.join('$PLUGIN_ROOT', 'assets', 'default_voice.wav')

def is_mlx_available():
    try:
        import mlx_audio
        return os.path.exists(MLX_VOICE_REF)
    except ImportError:
        return False

def speak_say(message):
    clean_message = re.sub(r'\[[\w\s]+\]\s*', '', message)
    subprocess.run(['say', '-v', SAY_VOICE, '-r', str(SAY_RATE), clean_message])

def speak_mlx(message):
    try:
        from mlx_server_utils import speak_mlx_http
        speak_mlx_http(message, speed=MLX_SPEED)
    except Exception as e:
        print(f'MLX TTS failed: {e}, using macOS say')
        speak_say(message)

def speak(message):
    if is_mlx_available():
        speak_mlx(message)
    else:
        speak_say(message)

speak('''$TEXT''')
"
