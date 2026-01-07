---
description: Summarize long text and speak the summary
---

Summarize long text and speak a condensed version. This command is **fire-and-forget** - it returns immediately while summarization and audio happen in the background.

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/summary-say.sh $ARGUMENTS
```

You can call this and immediately proceed to show questions or other UI - the summarization and TTS audio will happen asynchronously.
