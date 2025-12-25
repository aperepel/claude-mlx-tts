# Troubleshooting

## Issues after plugin update?

The plugin caches a virtual environment with dependencies. After updates, you may need to wipe it:

```bash
rm -rf ~/.claude/plugins/aperepel__claude-mlx-tts/.venv
```

The venv will be recreated automatically on next use.

## No sound?

Test that MLX audio is working:

```bash
uv run --with mlx-audio python -c "import mlx_audio; print('OK')"
```

Check the log file for errors:

```bash
cat ~/.claude/plugins/aperepel__claude-mlx-tts/logs/tts-notify.log
```

## Model download issues?

Pre-download the model (~4GB) manually:

```bash
uv run --with huggingface-hub python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/chatterbox-turbo-fp16')"
```

## Hook not triggering?

The TTS only triggers when:
- Response took 15+ seconds, OR
- 2+ tool calls were made, OR
- Your message contained "think" keywords (think, ultrathink, etc.)

Check the log to see threshold decisions:

```bash
tail -20 ~/.claude/plugins/aperepel__claude-mlx-tts/logs/tts-notify.log
```

## Fallback to macOS say?

If MLX fails, the plugin falls back to macOS `say` command. Check logs for the failure reason. Common causes:
- Missing voice reference file (`assets/default_voice.wav`)
- Corrupted model cache (wipe `~/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo-fp16`)
