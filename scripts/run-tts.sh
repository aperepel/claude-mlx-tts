#!/bin/bash
# Wrapper script for TTS hook - ensures venv is ready and runs the Python script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PLUGIN_ROOT/.venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" 2>/dev/null
    uv pip install --python "$VENV_DIR/bin/python" mlx-audio librosa mlx-lm einops sounddevice pyloudnorm 2>/dev/null
fi

# Run the TTS script with the venv's Python
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/tts-notify.py"
