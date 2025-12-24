# Claude Summary TTS

Text-to-speech notifications when Claude Code finishes deep work.

When Claude completes a task that took significant effort, this plugin speaks a brief summary so you know it's ready for review - useful when you've stepped away from the screen.

## Features

- **Smart triggers** - Only notifies on deep work (long responses, multiple tool calls, or thinking mode)
- **Concise summaries** - Uses Claude to generate 10-15 word spoken summaries
- **Zero dependencies** - Uses macOS `say` command by default
- **Optional voice cloning** - Upgrade to MLX for custom voice synthesis

## Requirements

- macOS (uses `say` command)
- Claude Code installed and authenticated

## Installation

```bash
/plugin install github:aperepel/claude-summary-tts
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

For custom voice synthesis on Apple Silicon:

1. Install dependencies:
```bash
pip install mlx-audio soundfile sounddevice librosa
```

2. Download a model (e.g., via LM Studio):
```
mlx-community/chatterbox-turbo-fp16
```

3. Create a 10-20 second voice reference WAV file

4. Edit `scripts/tts-notify.py`:
```python
MLX_MODEL_PATH = "~/.lmstudio/models/mlx-community/chatterbox-turbo-fp16"
MLX_VOICE_REF = "~/.claude/hooks/voice_ref.wav"
MLX_SPEED = 1.6
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
- Ensure model path and voice ref are correct
- Falls back to `say` automatically on error

## License

MIT
