#!/bin/bash
# Summarize text and speak it
# Usage: ./summary-say.sh <long text>

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
    echo "Usage: summary-say <long text to summarize and speak>"
    exit 1
fi

exec "$PYTHON" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')

import subprocess
import os
import re

SAY_VOICE = 'Daniel'
SAY_RATE = 180
MLX_SPEED = 1.6
MLX_VOICE_REF = os.path.join('$PLUGIN_ROOT', 'assets', 'default_voice.wav')
ATTENTION_PREFIX = '[clear throat] Attention on deck.'

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

def summarize(text):
    prompt = f'Convert to ONE short spoken sentence (max 15 words). No intro, no quotes, just the summary:\n\n{text[:1500]}'
    tmp_dir = '/tmp/claude-tts-tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    env = os.environ.copy()
    env['TMPDIR'] = tmp_dir
    try:
        result = subprocess.run(
            ['claude', '-p', '--settings', '{\"hooks\":{},\"alwaysThinkingEnabled\":false}', '--no-session-persistence', prompt],
            capture_output=True, text=True, timeout=30, env=env
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return 'Claude has completed its task.'

text = '''$TEXT'''
print('Summarizing...')
summary = summarize(text)
print(f'Summary: {summary}')
message = f'{ATTENTION_PREFIX} ... {summary}'
speak(message)
"
