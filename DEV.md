# Development Guide

## Setup

```bash
# Clone the repo
git clone https://github.com/aperepel/claude-mlx-tts.git
cd claude-mlx-tts
```

## Uninstall Marketplace Version (if installed)

```bash
claude plugin uninstall claude-mlx-tts
```

## Run from Local Checkout

Capture the checkout directory and use it to start Claude:

```bash
cd claude-mlx-tts
PLUGIN_DIR=$(pwd)
claude --plugin-dir "$PLUGIN_DIR"
```

This ensures the plugin loads from your local checkout regardless of where you run Claude from later.

## Iteration Workflow

1. **Edit code** - modify `scripts/tts-notify.py`
2. **Restart Claude** - changes take effect on new session
3. **Trigger TTS** - use `think about...` to test

No build step needed. The hook runs your local script directly.

## Test TTS Manually

```bash
# Test the detection logic
echo '{"transcript_path": "/path/to/transcript.jsonl"}' | \
  uv run --with mlx-audio --isolated python scripts/tts-notify.py

# Test MLX audio directly
uv run --with mlx-audio --isolated python -c "
import mlx_audio
print('MLX audio loaded OK')
"

# Test voice cloning
uv run --with mlx-audio --isolated python -m mlx_audio.tts.generate \
  --model mlx-community/chatterbox-turbo-fp16 \
  --text 'Testing voice cloning' \
  --ref_audio assets/default_voice.wav \
  --ref_text '.' \
  --speed 1.6 \
  --play
```

## Debug Hook Execution

Add print statements to `scripts/tts-notify.py` - output goes to Claude's hook log.

Or run the full pipeline manually:

```bash
# Find latest transcript
TRANSCRIPT=$(ls -t ~/.claude/projects/*/agent-*.jsonl 2>/dev/null | head -1)

# Run hook with it
echo "{\"transcript_path\": \"$TRANSCRIPT\"}" | \
  uv run --with mlx-audio --isolated python scripts/tts-notify.py
```

## Project Structure

```
claude-mlx-tts/
├── .claude-plugin/
│   ├── plugin.json        # Plugin metadata
│   └── marketplace.json   # Marketplace listing
├── hooks/
│   └── hooks.json         # Hook registration (Stop hook)
├── scripts/
│   └── tts-notify.py      # Main TTS logic
├── assets/
│   └── default_voice.wav  # Bundled voice reference
├── README.md              # User docs
├── RECORDING.md           # Custom voice guide
└── DEV.md                 # This file
```

## Key Files

- **hooks/hooks.json** - Registers the Stop hook, runs on every Claude response
- **scripts/tts-notify.py** - All logic: threshold detection, summarization, TTS
- **assets/default_voice.wav** - Bundled voice for zero-config experience

## Configuration

Edit `scripts/tts-notify.py`:

```python
MIN_DURATION_SECS = 15      # Lower for testing (e.g., 5)
MIN_TOOL_CALLS = 2          # Lower for testing (e.g., 1)
```

## Publishing Changes

```bash
git add -A
git commit -m "Description of changes"
git push
```

Users update via:
```bash
cd ~/.claude/plugins/marketplaces/claude-mlx-tts
git pull
```
