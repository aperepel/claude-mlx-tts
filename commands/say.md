---
description: Speak text directly using TTS (smoke test)
---

Speak text using TTS. This command is **fire-and-forget** - it returns immediately while audio plays in the background.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/say.sh $ARGUMENTS
```

You can call this and immediately proceed to show questions or other UI - the TTS audio will play asynchronously.
