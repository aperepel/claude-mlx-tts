# Migrating Plugin Skills to Markdown Commands

Guide for converting JSON-based plugin "skills" to faster markdown slash commands.

## Why Migrate?

| JSON Skills (Old) | Markdown Commands (New) |
|-------------------|------------------------|
| Claude interprets instruction, calls Bash | Inline `!` backtick executes directly |
| Permission prompt on each run | `allowed-tools` pre-authorizes |
| Multiple round trips | Single prompt expansion |
| Slower | **Faster** |

## Prerequisites

- Claude Code 2.1.0+
- Add `"minClaudeVersion": "2.1.0"` to `.claude-plugin/plugin.json`

## Directory Structure

```
your-plugin/
├── .claude-plugin/
│   └── plugin.json          # Add minClaudeVersion
├── commands/                 # NEW: Markdown commands
│   ├── your-command.md
│   └── another-command.md
├── hooks/
│   └── hooks.json           # Keep hooks, remove "skills" section
└── scripts/
    └── your-script.sh
```

## Command File Format

Create `commands/<command-name>.md`:

```markdown
---
allowed-tools: Bash(*)
description: Short description shown in /help
---

Brief instruction for what to do.

!`${CLAUDE_PLUGIN_ROOT}/scripts/your-script.sh $ARGUMENTS`
```

### Key Elements

1. **`allowed-tools: Bash(*)`** - Pre-authorizes Bash, no permission prompts
2. **`description:`** - Shows in `/help` and autocomplete
3. **`!` backtick syntax** - Executes inline during prompt expansion
4. **`${CLAUDE_PLUGIN_ROOT}`** - Resolves to plugin directory
5. **`$ARGUMENTS`** - User's arguments after the command name

## Migration Steps

### 1. Create commands directory

```bash
mkdir -p commands
```

### 2. Convert each JSON skill to markdown

**Before** (in `hooks/hooks.json`):
```json
{
  "skills": {
    "my-command": {
      "description": "Do something",
      "instruction": "Run the script to do something",
      "command": "${CLAUDE_PLUGIN_ROOT}/scripts/do-something.sh"
    }
  }
}
```

**After** (in `commands/my-command.md`):
```markdown
---
allowed-tools: Bash(*)
description: Do something
---

Run this command:

!`${CLAUDE_PLUGIN_ROOT}/scripts/do-something.sh $ARGUMENTS`
```

### 3. Update plugin.json

```json
{
  "name": "your-plugin",
  "version": "X.Y.Z",
  "minClaudeVersion": "2.1.0",
  ...
}
```

### 4. Remove skills from hooks.json

Keep the `"hooks"` section, delete the `"skills"` section.

### 5. Test

```bash
claude --plugin-dir /path/to/your-plugin
# Type /my-command and verify it works
```

## Advanced Features

### Commands with context access

Use `context: fork` to access conversation history:

```markdown
---
allowed-tools: Bash(*)
description: Summarize and act on last response
context: fork
---

Look at the last assistant response and summarize it, then:

!`${CLAUDE_PLUGIN_ROOT}/scripts/speak.sh "Summary: [your summary]"`
```

### Multiple tool permissions

```markdown
---
allowed-tools: Bash(git *), Bash(npm *), Read(*), Edit(*)
description: Complex workflow command
---
```

### Specific command patterns

```markdown
---
allowed-tools: Bash(git add:*), Bash(git commit:*)
description: Git commit helper
---
```

## Troubleshooting

### Command not appearing in /help
- Verify file is in `commands/` directory (not `.claude-plugin/commands/`)
- Check frontmatter has `description:` field
- Restart Claude Code session

### Permission still prompted
- Ensure `allowed-tools:` matches the commands being run
- Use `Bash(*)` for broad authorization

### ${CLAUDE_PLUGIN_ROOT} not resolving
- Verify running with `--plugin-dir` flag
- Check plugin.json is valid JSON

## Reference

- [Claude Code Slash Commands](https://code.claude.com/docs/en/slash-commands)
- [Plugin Structure](https://github.com/anthropics/claude-code/blob/main/plugins/README.md)
- [Example: commit-commands plugin](https://github.com/anthropics/claude-code/tree/main/plugins/commit-commands)
