#!/usr/bin/env python3
"""Auto-approve TTS-related bash commands for the plugin."""
import json
import sys

try:
    hook_input = json.load(sys.stdin)
except Exception:
    sys.exit(1)

tool_input = hook_input.get("tool_input", {})
command = tool_input.get("command", "")

# Auto-approve TTS-related commands
TTS_SCRIPTS = ["tts-init.sh", "tts-start.sh", "tts-stop.sh", "tts-status.sh", "tts-config.sh", "run-tts.sh", "say.sh", "summary-say.sh"]
if any(script in command for script in TTS_SCRIPTS):
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PermissionRequest",
            "decision": {"behavior": "allow"}
        }
    }
    print(json.dumps(output))
    sys.exit(0)

# Don't handle other commands (let normal permission flow happen)
sys.exit(1)
