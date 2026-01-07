#!/bin/bash
# Summarize text and speak it (fire-and-forget)
# Usage: ./summary-say.sh <long text>
#
# This script forks TTS to background and returns immediately.
# This allows callers to continue working while audio plays.

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

# Fork TTS to background and return immediately
# This makes the script fire-and-forget so callers don't block
"$PYTHON" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')

import subprocess
import os
import re
import logging

# Configure logging so mlx_tts_core metrics are visible
logging.basicConfig(level=logging.INFO, format='%(message)s')

SAY_VOICE = 'Daniel'
SAY_RATE = 180
ATTENTION_PREFIX = '[clear throat] Attention on deck.'

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
" >/dev/null 2>&1 &

# Disown the background process so it's not tied to this shell
disown 2>/dev/null

echo "TTS summarizing started"
