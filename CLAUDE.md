# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Summary TTS is a Claude Code plugin that provides text-to-speech notifications when Claude finishes "deep work" (long responses, multiple tool calls, or thinking mode). It uses macOS `say` command by default with optional MLX voice cloning.

## Architecture

- **Plugin structure**: Uses `.claude-plugin/plugin.json` for plugin metadata and `hooks/hooks.json` for hook registration
- **Single script**: `scripts/tts-notify.py` contains all logic - threshold checks, summarization, and TTS playback
- **Hook type**: Registers as a "Stop" hook that fires when Claude finishes responding
- **Summarization**: Calls `claude -p` subprocess to generate brief spoken summaries
- **MLX auto-detection**: Checks if `mlx_audio` is importable and voice reference exists

## Development Commands

```bash
# Install plugin locally for development
claude --plugin-dir ~/projects/claude-summary-tts

# Install all dependencies for development (MLX + pytest)
uv sync --extra mlx --extra dev

# Test macOS say command
say -v Daniel -r 180 "Test message"

# Pre-download MLX model (~4GB for fp16) - optional, auto-downloads on first use
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/chatterbox-turbo-fp16')"
```

## Release Process

**CRITICAL**: Versions must stay synchronized across three files:

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `.claude-plugin/plugin.json` | `"version": "X.Y.Z"` |
| `.claude-plugin/marketplace.json` | `metadata.version` AND `plugins[0].version` |

Release workflow:
```bash
# 1. Update version in all three files
# 2. Commit: git commit -m "Release vX.Y.Z: [description]"
# 3. Tag: git tag -a vX.Y.Z -m "Version X.Y.Z"
# 4. Push: git push origin main && git push origin vX.Y.Z
```

Note: Git tags are for release history only. The marketplace always pulls the latest commit from main—users update when they run `/plugin marketplace update` followed by `/plugin update`.

## Key Configuration (in scripts/tts-notify.py)

- `MIN_DURATION_SECS`: Response duration threshold (default 15s)
- `MIN_TOOL_CALLS`: Tool call count threshold (default 2)
- `THINKING_KEYWORDS`: Keywords that trigger TTS regardless of duration
- `SAY_VOICE` / `SAY_RATE`: macOS voice settings
- `MLX_MODEL`: HuggingFace model ID (auto-downloads on first use)
- `MLX_VOICE_REF`: Path to voice reference WAV file
