# Claude MLX TTS

Voice-cloned TTS notifications for Claude Code using [Chatterbox Turbo](https://www.resemble.ai/chatterbox/) on Apple Silicon.

When Claude finishes deep work, hear a brief summary in your own cloned voice—so you know it's ready without watching the screen.

## Features

- **Voice cloning** - Clone any voice from a 10-20 second sample using MLX Chatterbox Turbo
- **Paralinguistic tags** - Natural expressions like `[clear throat]`, `[laugh]`, `[sigh]`
- **Smart triggers** - Only notifies on deep work (long responses, multiple tool calls, or thinking mode)
- **Attention prefix** - "Attention on deck" heads-up before the summary
- **Fallback** - Uses macOS `say` if MLX not configured

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Claude Code installed and authenticated
- ~4GB disk space for the model

## Installation

```bash
# Install the plugin
claude plugin install aperepel/claude-mlx-tts

# Navigate to plugin directory and install MLX dependencies
cd ~/.claude/plugins/claude-mlx-tts
uv sync --extra mlx
```

## Voice Reference Setup

Record a voice sample for cloning:

| Requirement | Specification |
|-------------|---------------|
| **Duration** | 10-20 seconds of speech |
| **Format** | WAV (not MP3/M4A/AAC) |
| **Sample rate** | 24kHz or higher |
| **Channels** | Mono preferred |
| **Content** | Single speaker, no background noise |
| **Speech** | Clear articulation, natural pace |

```bash
# Create config directory
mkdir -p ~/.config/claude-tts

# Record using sox (install: brew install sox)
rec -r 24000 -c 1 ~/.config/claude-tts/voice_ref.wav trim 0 20

# Or convert from existing audio (Voice Memos, etc.)
ffmpeg -i input.m4a -ar 24000 -ac 1 -t 20 ~/.config/claude-tts/voice_ref.wav
```

**Tips:**
- Read naturally, as if explaining something to a colleague
- Record in a quiet room (closets work great for dampening echo)
- Avoid whispering, shouting, or exaggerated emotions

## Model Options

The model downloads automatically on first use. Available quantizations:

| Model | Size | Quality |
|-------|------|---------|
| `chatterbox-turbo-fp16` | ~4GB | Best (default) |
| `chatterbox-turbo-8bit` | ~1GB | Great |
| `chatterbox-turbo-6bit` | ~750MB | Good |
| `chatterbox-turbo-4bit` | ~500MB | Acceptable |

To switch models, edit `scripts/tts-notify.py`:

```python
MLX_MODEL = "mlx-community/chatterbox-turbo-8bit"
```

## Configuration

Edit `scripts/tts-notify.py`:

```python
# Trigger thresholds
MIN_DURATION_SECS = 15      # Response duration
MIN_TOOL_CALLS = 2          # Tool call count

# Attention prefix (uses Chatterbox paralinguistic tags)
ATTENTION_PREFIX = "[clear throat] Attention on deck."

# Voice cloning
MLX_MODEL = "mlx-community/chatterbox-turbo-fp16"
MLX_VOICE_REF = "~/.config/claude-tts/voice_ref.wav"
MLX_SPEED = 1.6
```

## Trigger Conditions

TTS fires when ANY condition is met:
- Response took >= 15 seconds
- >= 2 tool calls were made
- User message contained `think`, `think hard`, or `ultrathink`

Quick responses stay silent.

## How It Works

1. **Stop hook fires** when Claude finishes responding
2. **Threshold check** determines if notification is warranted
3. **Summarization** via `claude -p` generates a 10-15 word summary
4. **TTS playback** via MLX Chatterbox (or macOS `say` as fallback)

## Troubleshooting

**No voice cloning?**
```bash
# Check voice reference exists
ls ~/.config/claude-tts/voice_ref.wav

# Check MLX is installed
python -c "import mlx_audio; print('OK')"
```

**Model download issues?**
```bash
# Manual download
hf download mlx-community/chatterbox-turbo-fp16

# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo-*
```

**Falls back to macOS say?**
- Voice reference file missing or wrong format
- MLX dependencies not installed
- Check the requirements table above

## License

MIT
