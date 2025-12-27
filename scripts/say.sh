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
import logging

# Configure logging so mlx_tts_core metrics are visible
logging.basicConfig(level=logging.INFO, format='%(message)s')

SAY_VOICE = 'Daniel'
SAY_RATE = 180

def is_mlx_available():
    try:
        import mlx_audio
        from tts_config import discover_voices
        return len(discover_voices()) > 0
    except ImportError:
        return False

def speak_say(message):
    clean_message = re.sub(r'\[[\w\s]+\]\s*', '', message)
    subprocess.run(['say', '-v', SAY_VOICE, '-r', str(SAY_RATE), clean_message])

def speak_mlx(message):
    try:
        from mlx_server_utils import speak_mlx_http
        speak_mlx_http(message)  # Uses speed from config
    except Exception as e:
        print(f'HTTP TTS failed: {e}, trying direct API')
        try:
            from mlx_tts_core import speak_mlx as speak_mlx_direct
            speak_mlx_direct(message)  # Uses mlx_tts_core with metrics
        except Exception as e2:
            print(f'Direct TTS failed: {e2}, using macOS say')
            speak_say(message)

def speak(message):
    if is_mlx_available():
        speak_mlx(message)
    else:
        speak_say(message)

speak('''$TEXT''')
"
