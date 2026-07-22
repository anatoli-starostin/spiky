#!/usr/bin/env python3
"""Single source of truth for this host's PAIRED AGENT UNITS.

Imported by BOTH hooks that arm them, so the two can never drift apart:
  * session_start_bridge.py  (SessionStart)     -- fires when a session starts
  * ensure_listener.py       (UserPromptSubmit) -- fires on EVERY prompt, in every
    session type. This is the reliable net: SessionStart's behaviour on
    `claude --resume` is undocumented, so it cannot be trusted alone.

That gap is exactly how nebius-h100 came back with ONLY its Telegram bridge armed:
ensure_listener was the net that actually fired, and it knew about the listener but
nothing about the Slack units. Hence this shared list.

Each unit is a session-tied Monitor: it lives and dies with the Claude session (the
body). That IS the pairing -- when the session or the host goes down, the whole
agent, face included, goes with it.
"""
import os
import subprocess

FACADE = "/home/astarostin/work/slack-facade"

# (label, pgrep pattern, Monitor command)
# NOTE (2026-07-17): Telegram is DEPRECATED fleet-wide — approvals and chat moved to
# Slack (DM approvals via the consciousness). The inbound listener is no longer armed.
# To re-enable, restore the entry below.
UNITS = [
    # ("Telegram inbound listener (Anatoly's phone bridge)",
    #  "agent_bridge/tg_monitor.py",
    #  "python3 -u /home/astarostin/.claude/agent_bridge/tg_monitor.py"),
    ("Slack body-watch (delegated-task ears)",
     "body_bridge.py watch",
     f"cd {FACADE} && python3 body_bridge.py watch"),
    ("Slack consciousness (the paired Slack face)",
     "python -u app.py",
     f"cd {FACADE} && .venv/bin/python -u app.py"),
]


def alive(pattern: str) -> bool:
    """Exact process check. NOT a heartbeat file -- that can still look fresh for
    ~90s after a session exit killed the old Monitor, giving a false 'already
    running' right after a restart. No process = arm it."""
    try:
        r = subprocess.run(["pgrep", "-f", pattern],
                           capture_output=True, text=True, timeout=5)
        return bool(r.stdout.strip())
    except Exception:
        return False


def missing():
    """[(label, command)] for every unit that should be running here but isn't.
    Slack units are only claimed on hosts that actually have the facade installed."""
    out = []
    for label, pattern, cmd in UNITS:
        if not label.startswith("Telegram") and not os.path.isdir(FACADE):
            continue  # no slack-facade on this host
        if not alive(pattern):
            out.append((label, cmd))
    return out


def arm_instruction():
    """The context to inject, or None when everything is already up."""
    gaps = missing()
    if not gaps:
        return None
    lines = [
        "Some paired agent processes are NOT running. As your FIRST actions this "
        "session, BEFORE replying to the user, arm each MISSING one below with the "
        "Monitor tool (persistent=true, timeout_ms=3600000). They are session-tied, so "
        "they pair with this session (the body) — when it ends, they end. Arm ALL of "
        "them, not just the first: skipping any leaves Anatoly unable to reach this "
        "machine on that channel."
    ]
    for i, (label, cmd) in enumerate(gaps, 1):
        lines.append(f'{i}. {label}: Monitor command "{cmd}"')
    return "\n".join(lines)
