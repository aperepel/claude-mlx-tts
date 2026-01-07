# Development Guide

For system architecture and component diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

```bash
# Clone the repo
git clone https://github.com/aperepel/claude-mlx-tts.git
cd claude-mlx-tts

# Install all dependencies (MLX + dev tools)
uv sync --extra mlx --extra dev
```

## Uninstall Marketplace Version

If you have the marketplace version installed, uninstall it first:

```bash
claude plugin uninstall claude-mlx-tts
```

## Run from Local Checkout

```bash
cd claude-mlx-tts
claude --plugin-dir "$(pwd)"
```

## Iteration Workflow

1. Edit code in `scripts/`
2. Restart Claude session
3. Test with `/say Hello` or trigger deep work

No build step needed.

## Cleanup

To switch back to the marketplace version:

```bash
# Stop any running TTS server
/tts-stop

# Exit Claude, then reinstall from marketplace
claude plugin install aperepel/claude-mlx-tts
```

To fully remove local dev artifacts:

```bash
# Remove venv and logs
rm -rf .venv logs/

# Remove cached model (optional, ~4GB)
rm -rf ~/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo*
```
