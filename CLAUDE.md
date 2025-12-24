# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Summary TTS is a Claude Code plugin that provides text-to-speech notifications when Claude finishes "deep work" (long responses, multiple tool calls, or thinking mode). It uses macOS `say` command by default with optional MLX voice cloning.

## Architecture

- **Plugin structure**: Uses `.claude-plugin/plugin.json` for plugin metadata and `hooks/hooks.json` for hook registration
- **Single script**: `scripts/tts-notify.py` contains all logic - threshold checks, summarization, and TTS playback
- **Hook type**: Registers as a "Stop" hook that fires when Claude finishes responding
- **Summarization**: Calls `claude -p` subprocess to generate brief spoken summaries

## Running/Testing

Test the TTS script manually (requires JSON input on stdin):
```bash
echo '{"transcript_path": "/path/to/transcript.jsonl"}' | python3 scripts/tts-notify.py
```

Test macOS say command:
```bash
say -v Daniel -r 180 "Test message"
```

Install plugin locally for development:
```bash
claude --plugin-dir ~/projects/claude-summary-tts
```

## Key Configuration (in scripts/tts-notify.py)

- `MIN_DURATION_SECS`: Response duration threshold (default 15s)
- `MIN_TOOL_CALLS`: Tool call count threshold (default 2)
- `THINKING_KEYWORDS`: Keywords that trigger TTS regardless of duration
- `SAY_VOICE` / `SAY_RATE`: macOS voice settings
- `MLX_MODEL_PATH` / `MLX_VOICE_REF`: Optional MLX voice cloning paths
