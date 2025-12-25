# Architecture

## Overview

Claude MLX TTS is a Claude Code plugin that provides voice notifications when Claude finishes "deep work". It uses a tiered TTS system with multiple fallbacks for reliability.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Claude Code                                  │
│                                                                      │
│  User prompt ──▶ Claude responds ──▶ Stop hook fires                │
│                                            │                         │
└────────────────────────────────────────────┼─────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      tts-notify.py                                   │
│                                                                      │
│  1. Check thresholds (duration, tool calls, thinking keywords)      │
│  2. Summarize response via `claude -p`                              │
│  3. Speak summary via TTS backend                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## TTS Backend Fallback Chain

The plugin tries backends in order, falling back on failure:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│   │  MLX HTTP    │────▶│  MLX Direct  │────▶│  macOS say   │        │
│   │  Server      │fail │  API         │fail │  command     │        │
│   └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                    │                    │                  │
│         │                    │                    │                  │
│       fast              ~10s cold              quick                │
│   (pre-warmed)      (reloads model each time) (low quality)         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

| Backend | Latency | Voice Cloning | Requirements |
|---------|---------|---------------|--------------|
| MLX HTTP Server | Fast (pre-warmed) | Yes | `/tts-start` to pre-warm |
| MLX Direct API | ~10s (reloads model each time) | Yes | MLX deps installed |
| macOS `say` | Quick (low quality) | No | None (built-in) |

## MLX Server Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           /tts-init                                  │
│                      (one-time setup)                                │
│                                                                      │
│   1. Creates .venv and installs MLX dependencies (uv sync)          │
│   2. Downloads model from HuggingFace Hub (~4GB)                    │
│                                                                      │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Local Storage                                 │
│                                                                      │
│  Plugin venv:   ~/.claude/plugins/claude-mlx-tts/.venv/             │
│  Model cache:   ~/.cache/huggingface/hub/models--mlx-community--*   │
│                                                                      │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                         /tts-start │  (loads into GPU memory)
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     mlx_audio.server                                 │
│                     (localhost:21099)                                │
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   FastAPI   │───▶│  Chatterbox │───▶│   Audio     │              │
│  │   /v1/...   │    │   Model     │    │   Output    │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                                      │
│  GPU Memory: ~4GB when loaded                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                         /tts-stop  │  (unloads, reclaims memory)
                                    ▼
                              (server killed)
```

## Hook Flow

```
Claude Code Stop Event
         │
         ▼
┌─────────────────────┐
│   hooks.json        │
│   "Stop" hook       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   run-tts.sh        │──────▶ Creates .venv if needed
└──────────┬──────────┘        (lazy initialization)
           │
           ▼
┌─────────────────────┐
│   tts-notify.py     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     No
│ Duration >= 15s?    │─────────┐
│ OR tools >= 2?      │         │
│ OR thinking keyword?│         │
└──────────┬──────────┘         │
           │ Yes                │
           ▼                    │
┌─────────────────────┐         │
│ Summarize via       │         │
│ `claude -p`         │         │
└──────────┬──────────┘         │
           │                    │
           ▼                    │
┌─────────────────────┐         │
│ speak()             │         │
│ (backend chain)     │         │
└──────────┬──────────┘         │
           │                    │
           ▼                    ▼
        [audio]              (silent)
```

## File Structure

```
claude-mlx-tts/
├── .claude-plugin/
│   ├── plugin.json          # Plugin metadata
│   └── marketplace.json     # Marketplace listing
│
├── assets/
│   └── default_voice.wav    # Bundled voice reference
│
├── commands/                # Slash command definitions
│   ├── tts-init.md
│   ├── tts-start.md
│   ├── tts-stop.md
│   └── tts-status.md
│
├── hooks/
│   ├── hooks.json           # Hook + skill registration
│   └── approve-tts.py       # Auto-approve TTS commands
│
├── scripts/
│   ├── tts-notify.py        # Main TTS logic (Stop hook)
│   ├── mlx_server_utils.py  # Server lifecycle management
│   ├── tts-init.sh          # Install MLX deps + download model
│   ├── tts-start.sh         # Start HTTP server
│   ├── tts-stop.sh          # Stop HTTP server
│   ├── tts-status.sh        # Check server status
│   └── run-tts.sh           # Hook entry point (lazy init)
│
└── logs/
    ├── tts-notify.log       # TTS hook logs
    └── mlx-server.log       # Server logs
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_USE_HTTP` | `true` | Use HTTP server (vs direct API) |
| `TTS_SERVER_PORT` | `21099` | Server port |
| `TTS_SERVER_HOST` | `localhost` | Server host |
| `TTS_START_TIMEOUT` | `60` | Server startup timeout (seconds) |
| `MLX_TTS_MODEL` | `mlx-community/chatterbox-turbo-fp16` | HuggingFace model ID |
| `MLX_TTS_SPEED` | `1.6` | Speech speed multiplier |
