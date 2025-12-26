#!/bin/bash
# Initialize MLX TTS dependencies
# Usage: ./tts-init.sh
#
# Installs MLX voice cloning dependencies (~4GB model download on first use).
# Safe to run multiple times - uv sync is idempotent.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PLUGIN_ROOT"

# Unset any active venv to avoid uv warnings
unset VIRTUAL_ENV

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install it with: brew install uv"
    echo "Or: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Installing MLX TTS dependencies..."
echo "This may take a minute on first run."
echo ""

# Run uv sync with MLX extras
if ! uv sync --extra mlx; then
    echo ""
    echo "Failed to install dependencies."
    echo "Check that you have enough disk space and try again."
    exit 1
fi

echo ""
echo "Downloading MLX model..."
echo "This is a one-time download."
echo ""

# Read model from tts-notify.py config
MLX_MODEL=$(grep -E "^MLX_MODEL\s*=" "$SCRIPT_DIR/tts-notify.py" | sed 's/.*"\(.*\)".*/\1/' || echo "mlx-community/chatterbox-turbo-fp16")
echo "Model: $MLX_MODEL"

# Download the model
if uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$MLX_MODEL')"; then
    echo ""
    echo "MLX TTS setup complete!"
    echo "Run /tts-start to start the server and cache voice embeddings."
else
    echo ""
    echo "Failed to download model."
    echo "Check your internet connection and try again."
    exit 1
fi
