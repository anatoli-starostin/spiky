#!/usr/bin/env python3
"""UserPromptSubmit hook: the approval channel FOLLOWS the request's origin.

No phone guard, no manual toggling needed. The rule is simply:
  * work delegated from Slack (a BODY_TASK from the consciousness) -> approvals
    are asked in Slack;
  * work typed at the console (Human Master at the keyboard) -> approvals are asked at
    the console.

It writes ~/.claude/slack_facade/approval_channel ("slack" | "console"), which the
permission gate reads to decide where to ask. Fires per turn, so the channel always
matches whatever kicked off the current turn. (A restart defaults it to "console" via
session_start_bridge; the face also pulls it to "slack" at delegate time -- this hook
is the per-turn refinement on top of both.)
"""
import sys, json, pathlib

CHANNEL = pathlib.Path.home() / ".claude" / "slack_facade" / "approval_channel"

def set_channel(v):
    try:
        CHANNEL.parent.mkdir(parents=True, exist_ok=True)
        CHANNEL.write_text(v)
    except Exception:
        pass

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    prompt = payload.get("prompt") or ""
    low = prompt.lower()
    if not prompt.strip():
        return  # empty/injected -> says nothing about origin, leave channel as-is
    # The consciousness (Slack face) is not the body; never let its session flip this.
    if "slack-facade/consciousness" in str(payload.get("cwd") or ""):
        return
    if "body_task" in low:
        set_channel("slack")     # Slack-delegated task -> approve in Slack
        return
    # Other injected notifications / config commands say nothing about origin.
    if ("<task-notification" in low or "tg_msg:" in low
            or "phone_guard" in low or "guard_on" in low):
        return
    set_channel("console")       # genuine console typing -> approve at the console

if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
