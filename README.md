# Claude MLX TTS

Voice-cloned TTS notifications for Claude Code using [Chatterbox Turbo](https://www.resemble.ai/chatterbox/) on Apple Silicon.

When Claude finishes deep work, hear a brief AI-generated summary spoken aloud—so you know it's ready without watching the screen.

## Features

- **AI-powered summaries** - Condenses Claude's response into a 10-15 word spoken update
- **Attention grabber** - "[clear throat] Attention on deck..." before each summary
- **Voice cloning** - Uses MLX Chatterbox Turbo (default voice included, or clone your own)
- **Smart detection** - Only triggers on deep work (15+ seconds, 2+ tool calls, or thinking mode)
- **Zero config** - Works out of the box with bundled voice

## Quick Start

### Install

First, add this repo as a plugin marketplace:

```bash
# From terminal
claude plugin marketplace add aperepel/claude-mlx-tts

# Or from within a Claude Code session
/plugin marketplace add aperepel/claude-mlx-tts
```

Then install the plugin:

```bash
# From terminal
claude plugin install claude-mlx-tts

# Or from within a Claude Code session
/plugin install claude-mlx-tts
```

### Enable Voice Cloning (MLX on Apple Silicon)

After installation, optionally enable MLX voice cloning:

```bash
# Install uv (if needed)
brew install uv        # or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then in a Claude Code session, run `/tts-init` to install dependencies and download the model.

> **Note:** The default fp16 model is ~4GB and may take a while to download. See [Model Options](#model-options) for smaller quantized alternatives.

> [Full uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

### Marketplace Management

```bash
/plugin marketplace list                  # View all marketplaces
/plugin marketplace update claude-mlx-tts # Refresh plugins
/plugin marketplace remove claude-mlx-tts # Remove marketplace
```

### Test It

Quick smoke test:

```
/say Hello world
```

Or trigger deep work to hear an automatic summary:

```
> think about what makes good API design
```

## Custom Voice

Want to use your own voice? See [RECORDING.md](RECORDING.md) for instructions on creating a custom voice reference.

Replace `assets/default_voice.wav` with your recording.

## Commands

| Command | Description |
|---------|-------------|
| `/tts-init` | Install MLX dependencies and download model (~4GB) |
| `/say <text>` | Speak text directly (smoke test) |
| `/summary-say <text>` | Summarize long text and speak it |
| `/tts-start` | Start server (pre-warms model for faster responses) |
| `/tts-stop` | Stop server (reclaims ~4GB GPU memory) |
| `/tts-status` | Check if server is running |

## Configuration

Edit `scripts/tts-notify.py`:

```python
MIN_DURATION_SECS = 15      # Response duration threshold
MIN_TOOL_CALLS = 2          # Tool call threshold
ATTENTION_PREFIX = "[clear throat] Attention on deck."
MLX_SPEED = 1.6             # Playback speed
MLX_MODEL = "mlx-community/chatterbox-turbo-fp16"  # Model to use
```

### Model Options

The default model is `chatterbox-turbo-fp16` (~4GB). For smaller memory footprint, try:

| Model | Size | Quality |
|-------|------|---------|
| `mlx-community/chatterbox-turbo-fp16` | ~4GB | Best |
| `mlx-community/chatterbox-turbo-8bit` | ~2GB | Good |
| `mlx-community/chatterbox-turbo-4bit` | ~1GB | Acceptable |

To use a different model, edit `MLX_MODEL` in `scripts/tts-notify.py` and run `/tts-init` to download it.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

## Requirements

**Basic (macOS `say`):**
- macOS (any Mac)
- Claude Code installed

**Voice Cloning (MLX):**
- macOS on Apple Silicon (M1/M2/M3/M4)
- Claude Code installed
- ~4GB disk space for the model

## License

MIT
