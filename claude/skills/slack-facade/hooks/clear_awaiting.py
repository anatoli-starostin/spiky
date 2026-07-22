#!/usr/bin/env python3
"""PostToolUse hook: the tool has RUN, so whatever approval was pending is resolved.

Counterpart to telegram_permission.py, which publishes
`~/.claude/agent_bridge/awaiting_approval` (contents: "console" or "telegram")
while a human is being asked. The Slack face reads that via check_status so it can
truthfully say "there's an approval waiting for you in the console / on Telegram"
instead of guessing. Without this, the flag would linger after approval and the face
would keep claiming it's blocked.
"""
import os
import pathlib
import time

AWAITING = pathlib.Path(os.path.expanduser("~/.claude/agent_bridge/awaiting_approval"))
TRACE = pathlib.Path(os.path.expanduser("~/.claude/agent_bridge/awaiting_trace.log"))

try:
    if AWAITING.exists():
        with TRACE.open("a") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] PostToolUse CLEARED flag (tool ran)\n")
    AWAITING.unlink(missing_ok=True)
except Exception:
    pass
