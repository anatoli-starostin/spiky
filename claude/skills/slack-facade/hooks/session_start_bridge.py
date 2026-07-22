#!/usr/bin/env python3
"""SessionStart hook: arm this host's paired agent units at session start.

The unit list and the checks live in paired_units.py, shared with
ensure_listener.py (UserPromptSubmit) — which is the more reliable net, since
SessionStart's behaviour on `claude --resume` is undocumented. Keeping both on one
shared list is what stops them drifting: nebius-h100 once came back with only its
Telegram bridge armed because the two hooks knew different things.
"""
import json
import pathlib
import sys

sys.path.insert(0, "/home/astarostin/.claude/hooks")
import paired_units  # noqa: E402

APPROVAL_CHANNEL = pathlib.Path("/home/astarostin/.claude/slack_facade/approval_channel")


def reset_cord_to_console():
    """After a restart, Anatoly is AT the console (he just started this session), so
    approvals belong at the console -- never in Slack. Default the cord to "console" on
    every SessionStart; approval_channel_follow (UserPromptSubmit) then re-points it per
    turn -- a Slack-delegated BODY_TASK pulls it to "slack", console typing keeps it
    "console". Without this, a stale "slack" left over from a previous run sends this
    console session's approvals to Anatoly's Slack DM, where they sit and time out with
    nobody at the console to answer -- exactly what happened on a nebius restart."""
    try:
        APPROVAL_CHANNEL.parent.mkdir(parents=True, exist_ok=True)
        APPROVAL_CHANNEL.write_text("console")
    except Exception:
        pass


def main():
    reset_cord_to_console()
    ctx = paired_units.arm_instruction()
    if not ctx:
        # Derive the "all running" note from the ACTUAL unit list, so it can never
        # name a unit this host no longer has (gpustar dropped the Telegram listener).
        names = ", ".join(u[0].split(" (")[0] for u in paired_units.UNITS)
        ctx = f"Paired agent Monitors ({names}) all running — no action needed."
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": ctx,
        }
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
