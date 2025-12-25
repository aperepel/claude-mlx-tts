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

### Basic Install (macOS `say`)

```bash
claude plugin install aperepel/claude-mlx-tts
```

Works immediately with zero dependencies. Uses the built-in macOS `say` command.

### Voice Cloning (MLX on Apple Silicon)

```bash
# 1. Install uv (if needed)
brew install uv        # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install plugin
claude plugin install aperepel/claude-mlx-tts
```

Then in a Claude Code session, run `/tts-init` to install dependencies and download the model (~4GB).

> [Full uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

### Installing During a Session

Already in a Claude Code session? Use the `/plugin` command:

```
/plugin
```

This opens an interactive browser where you can search for and install plugins without leaving your session. Search for `mlx-tts` and select install.

For direct installation:

```
/plugin install aperepel/claude-mlx-tts
```

After installation, the plugin is active immediately—no restart required.

### Test It

Start a new Claude session and trigger deep work:

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
```

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
