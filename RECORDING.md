# Custom Voice Recording Guide

Clone any voice—your own, a colleague, a character, or any speaker you like.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Duration** | 10-20 seconds of speech |
| **Format** | WAV |
| **Sample rate** | 24kHz or higher |
| **Channels** | Mono |
| **Content** | Single speaker, no background noise |
| **Speech** | Clear articulation, natural pace |

## Recording Methods

### Option A: Record with sox

```bash
# Install sox if needed
brew install sox

# Record (speak naturally, press Ctrl+C when done)
rec -r 24000 -c 1 voice_ref.wav trim 0 20
```

### Option B: Convert existing audio

```bash
# From Voice Memos, podcast clip, or any audio file
ffmpeg -i input.m4a -ar 24000 -ac 1 -t 20 voice_ref.wav
```

### Option C: Use QuickTime Player

1. Open QuickTime Player
2. File → New Audio Recording
3. Record 10-20 seconds of speech
4. Save and convert with ffmpeg:
   ```bash
   ffmpeg -i recording.m4a -ar 24000 -ac 1 voice_ref.wav
   ```

## Tips for Best Results

- **Environment**: Record in a quiet room. Closets work great for dampening echo.
- **Content**: Read a paragraph naturally, as if explaining something to a colleague.
- **Tone**: Avoid whispering, shouting, or exaggerated emotions.
- **Quality**: Use a decent microphone if available. Built-in Mac mic works but external is better.

## Installing Your Voice

Replace the bundled voice with your recording:

```bash
cp /path/to/your/voice_ref.wav ~/.claude/plugins/marketplaces/claude-mlx-tts/assets/default_voice.wav
```

## Model Options

If you want faster inference at the cost of quality, edit `scripts/tts-notify.py`:

```python
# Options (size/quality tradeoff):
MLX_MODEL = "mlx-community/chatterbox-turbo-fp16"   # ~4GB, best quality (default)
MLX_MODEL = "mlx-community/chatterbox-turbo-8bit"   # ~1GB, great quality
MLX_MODEL = "mlx-community/chatterbox-turbo-4bit"   # ~500MB, acceptable quality
```
