---
allowed-tools: Bash(*)
description: Summarize and speak the last response
context: fork
---

Look at the last assistant response in the conversation, create a one-sentence summary (max 15 words), then speak it:

!`${CLAUDE_PLUGIN_ROOT}/scripts/say.sh "Attention on deck... [your summary]"`
