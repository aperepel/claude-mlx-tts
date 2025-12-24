# Claude Summary TTS

Text-to-speech notifications when Claude Code finishes deep work.

When Claude completes a task that took significant effort, this plugin speaks a brief summary so you know it's ready for review - useful when you've stepped away from the screen.

## Features

- **Smart triggers** - Only notifies on deep work (long responses, multiple tool calls, or thinking mode)
- **Concise summaries** - Uses Claude to generate 10-15 word spoken summaries
- **Zero dependencies** - Uses macOS `say` command by default
- **Optional voice cloning** - Upgrade to MLX Chatterbox for custom voice synthesis

## Requirements

- macOS (uses `say` command)
- Claude Code installed and authenticated

## Installation

```bash
claude plugin install aperepel/claude-summary-tts
```

Or for local development:
```bash
claude --plugin-dir ~/projects/claude-summary-tts
```

## Configuration

Edit `scripts/tts-notify.py` to customize:

```python
# Thresholds for triggering TTS
MIN_DURATION_SECS = 15      # Response duration threshold
MIN_TOOL_CALLS = 2          # Tool call count threshold

# macOS 'say' settings
SAY_VOICE = "Daniel"        # Try: say -v ? to list voices
SAY_RATE = 180              # Words per minute
```

## Trigger Conditions

TTS fires when ANY of these conditions are met:
- Response took >= 15 seconds
- >= 2 tool calls were made
- User message contained thinking keywords (`think`, `think hard`, `ultrathink`)

Quick responses and simple queries stay silent.

## Optional: MLX Voice Cloning

For custom voice synthesis on Apple Silicon using [Chatterbox](https://www.resemble.ai/chatterbox/) (MIT licensed):

### 1. Install MLX dependencies

```bash
# Navigate to plugin directory
cd ~/.claude/plugins/summary-tts  # or your local dev path

# Install with uv (recommended)
uv sync --extra mlx

# Or with pip
pip install ".[mlx]"
```

### 2. Create a voice reference

Record a voice sample for cloning. Requirements:

| Requirement | Specification |
|-------------|---------------|
| **Duration** | 10-20 seconds of speech |
| **Format** | WAV (not MP3/M4A/AAC) |
| **Sample rate** | 24kHz or higher |
| **Channels** | Mono preferred (stereo auto-converted) |
| **Content** | Single speaker only |
| **Environment** | No background noise, music, or room reverb |
| **Speech** | Clear articulation, natural pace |

```bash
# Create config directory
mkdir -p ~/.config/claude-tts

# Record using sox (install: brew install sox)
rec -r 24000 -c 1 ~/.config/claude-tts/voice_ref.wav trim 0 20

# Or using ffmpeg from existing audio
ffmpeg -i input.m4a -ar 24000 -ac 1 -t 20 ~/.config/claude-tts/voice_ref.wav
```

**Tips for best results:**
- Read a paragraph naturally, as if explaining something to a colleague
- Avoid whispering, shouting, or exaggerated emotions
- Record in a quiet room (closets work great for dampening echo)
- If converting from Voice Memos, use the ffmpeg command above

### 3. First run downloads the model

The Chatterbox model (~4GB for fp16) downloads automatically on first use from HuggingFace. No manual download required.

### Configuration

```python
# In scripts/tts-notify.py
MLX_MODEL = "mlx-community/chatterbox-turbo-fp16"  # Model ID (auto-downloads)
MLX_VOICE_REF = "~/.config/claude-tts/voice_ref.wav"
MLX_SPEED = 1.6
```

### Choosing a Different Model

Available quantizations (trade-off: size/speed vs quality):

| Model | Size | Quality | Use Case |
|-------|------|---------|----------|
| `chatterbox-turbo-fp16` | ~4GB | Best | Default, highest fidelity |
| `chatterbox-turbo-8bit` | ~1GB | Great | Balanced size/quality |
| `chatterbox-turbo-6bit` | ~750MB | Good | Smaller footprint |
| `chatterbox-turbo-5bit` | ~600MB | Good | Even smaller |
| `chatterbox-turbo-4bit` | ~500MB | Acceptable | Fastest, smallest |

To switch models, edit `scripts/tts-notify.py`:

```python
MLX_MODEL = "mlx-community/chatterbox-turbo-8bit"  # Change to desired variant
```

The new model downloads automatically on next TTS playback. To pre-download or clear old models:

```bash
# Pre-download a specific model
huggingface-cli download mlx-community/chatterbox-turbo-8bit

# Clear HuggingFace cache (removes all cached models)
rm -rf ~/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo-*

# Or selectively remove one model
rm -rf ~/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo-fp16
```

## How It Works

1. **Stop hook fires** when Claude finishes responding
2. **Threshold check** determines if notification is warranted
3. **Summarization** via `claude -p` generates a brief spoken summary
4. **TTS playback** via `say` (or MLX if configured)

## Troubleshooting

**No sound?**
- Check `say "test"` works in terminal
- Verify Claude Code is authenticated (`claude --version`)

**Wrong triggers?**
- Adjust `MIN_DURATION_SECS` and `MIN_TOOL_CALLS` thresholds

**MLX not working?**
- Verify voice reference exists: `ls ~/.config/claude-tts/voice_ref.wav`
- Check MLX is installed: `python -c "import mlx_audio"`
- Falls back to `say` automatically on error

**Model download issues?**
- Check HuggingFace connectivity
- Manual download: `huggingface-cli download mlx-community/chatterbox-turbo-fp16`

## License

MIT
