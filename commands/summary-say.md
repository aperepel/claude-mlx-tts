---
allowed-tools: Bash(*)
description: Summarize and speak the last response
context: fork
---

# Summary-Say Command

Summarize the last assistant response and speak it aloud.

## Instructions

1. **Read the last assistant response** (the message immediately before this command was invoked)

2. **Create a substantive summary** (1-2 sentences, max 30 words):
   - **For interview endings**: Highlight 2-3 key decisions or outcomes, not just "interview complete"
   - **For spec generation**: Mention the main trade-offs or scope decisions
   - **For bead updates**: Summarize what was clarified or changed
   - Use natural, conversational language

3. **Speak it** using the Bash tool:
   ```
   ${CLAUDE_PLUGIN_ROOT}/scripts/say.sh "Attention on deck... <your substantive summary>"
   ```

## Examples

**Interview ending (GOOD - mentions decisions):**
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/say.sh "Attention on deck... We settled on JWT auth with refresh tokens, scoped to internal users only, targeting a two-week MVP."
```

**Interview ending (BAD - too generic):**
```bash
# DON'T do this:
${CLAUDE_PLUGIN_ROOT}/scripts/say.sh "Attention on deck... The interview is complete."
```

**Blitz completion:**
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/say.sh "Attention on deck... Clarified the caching strategy: Redis for sessions, in-memory for config. Added dependency on the auth task."
```

**Spec generation:**
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/say.sh "Attention on deck... Spec saved with twelve requirements. Key decisions: eventual consistency over strong, and mobile deferred to phase two."
```

## Key Principle

The summary should tell the listener what was **decided**, not just what **happened**. Extract the substance.
