#!/usr/bin/env python3
"""UserPromptSubmit hook: guarantee ALL the paired agent units are armed — even on
`claude --resume`, where SessionStart's firing is undocumented. UserPromptSubmit
fires on every prompt in every session type, so this is the RELIABLE net.

It used to know only about the Telegram listener. That's how nebius-h100 came back
from a restart with just its Telegram bridge running and no Slack face: SessionStart
either didn't fire or wasn't acted on, and the net that DID fire only nagged about
Telegram. The unit list now lives in paired_units.py, shared with
session_start_bridge.py, so the two can never drift apart again.

Injects nothing once everything is up, so it's silent in the normal case.
"""
import json
import sys

sys.path.insert(0, "/home/astarostin/.claude/hooks")
import paired_units  # noqa: E402


def main():
    ctx = paired_units.arm_instruction()
    if not ctx:
        return  # everything running — say nothing
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": ctx,
        }
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
