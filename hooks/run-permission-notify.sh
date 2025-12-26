#!/bin/bash
# Wrapper script for permission TTS hook - ensures venv is used for mlx_audio

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PLUGIN_ROOT/.venv"

# Use venv Python if it exists, otherwise fall back to system python
if [ -x "$VENV_DIR/bin/python" ]; then
    exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/permission-notify.py"
else
    exec python3 "$SCRIPT_DIR/permission-notify.py"
fi
