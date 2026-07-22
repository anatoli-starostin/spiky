#!/usr/bin/env python3
"""PermissionRequest hook: a permission dialog just APPEARED, so the agent is genuinely
awaiting a human decision at the console. Publish it (awaiting_approval=console) so the
Slack face flags its last message with ❗ and check_status can report it.

This DETECTS the real prompt instead of PREDICTING whether one will appear. The old
predictor (_host_allows in telegram_permission.py) guessed "Write is allowed -> no
prompt" and was wrong for a subagent writing to ~/.claude/skills (outside the project),
so the ❗ never showed. Detection has no such blind spot.

Cleared by clear_awaiting.py on PostToolUse (approved -> ran) / PermissionDenied
(denied) / Stop (turn end, catches cancel). Does not clobber a "telegram" marker that
the guard-on path may have written.
"""
import pathlib

AWAITING = pathlib.Path("/home/astarostin/.claude/agent_bridge/awaiting_approval")

try:
    cur = AWAITING.read_text().strip() if AWAITING.exists() else ""
    if cur != "telegram":
        AWAITING.write_text("console")
except Exception:
    pass
