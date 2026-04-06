#!/usr/bin/env python3
"""PreToolUse hook that blocks Claude from reading .env files."""

import json
import sys


def main():
    event = json.load(sys.stdin)
    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input", {})

    # Check Read, Edit, and Write tools for .env file access
    if tool_name in ("Read", "Edit", "Write"):
        file_path = tool_input.get("file_path", "")
        if file_path.endswith(".env") or "/.env" in file_path:
            result = {
                "decision": "block",
                "reason": "Access to .env files is blocked to protect secrets."
            }
            print(json.dumps(result))
            return

    # Check Bash tool for commands that might read .env
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if ".env" in command:
            result = {
                "decision": "block",
                "reason": "Access to .env files is blocked to protect secrets."
            }
            print(json.dumps(result))
            return

    # Allow everything else
    print(json.dumps({"decision": "approve"}))


if __name__ == "__main__":
    main()
