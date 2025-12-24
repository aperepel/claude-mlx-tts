# Installation Guide

## Step 1: Install the Plugin

```bash
claude plugin install aperepel/claude-mlx-tts
```

## Step 2: Install MLX Dependencies

```bash
cd ~/.claude/plugins/claude-mlx-tts
uv sync --extra mlx
```

> **Note:** If you don't have `uv`, install it first: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Step 3: Create Your Voice Reference

Create the config directory:

```bash
mkdir -p ~/.config/claude-tts
```

Record a 20-second voice sample using one of these methods:

**Option A: Record with sox**
```bash
# Install sox if needed
brew install sox

# Record (speak naturally for 20 seconds, then Ctrl+C)
rec -r 24000 -c 1 ~/.config/claude-tts/voice_ref.wav trim 0 20
```

**Option B: Convert existing audio**
```bash
# From Voice Memos, podcast clip, or any audio file
ffmpeg -i ~/path/to/your-voice.m4a -ar 24000 -ac 1 -t 20 ~/.config/claude-tts/voice_ref.wav
```

### Voice Reference Requirements

| Requirement | Specification |
|-------------|---------------|
| Duration | 10-20 seconds |
| Format | WAV |
| Sample rate | 24kHz or higher |
| Channels | Mono |
| Content | Single speaker, no background noise |

**Tips for best results:**
- Speak naturally, as if explaining something to a colleague
- Record in a quiet room
- Avoid whispering or exaggerated emotions

## Step 4: Verify Installation

Start a new Claude Code session:

```bash
claude
```

Trigger a deep work response:

```
> think about what makes a good commit message
```

After Claude responds, you should hear the TTS summary in your cloned voice.

## First Run

The Chatterbox model (~4GB) downloads automatically on the first TTS trigger. This is a one-time download.

To pre-download the model:

```bash
hf download mlx-community/chatterbox-turbo-fp16
```

## Troubleshooting

**No sound?**
```bash
# Check voice reference exists
ls ~/.config/claude-tts/voice_ref.wav

# Check MLX is installed
cd ~/.claude/plugins/claude-mlx-tts
uv run python -c "import mlx_audio; print('OK')"
```

**Falls back to macOS say?**
- Voice reference file missing or wrong format
- MLX dependencies not installed
- Run Step 2 again

**Model download fails?**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo-*
hf download mlx-community/chatterbox-turbo-fp16
```
