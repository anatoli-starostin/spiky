#!/usr/bin/env python3
"""Audit one replica's slack-facade wiring. Run on each host; prints a report."""
import json
import os
import pathlib
import re
import socket
import subprocess

HOME = pathlib.Path.home()
# Where the facade is deployed — outside the sbox cage's writable zone (see the
# agent-cage SKILL). Matches cage_policy's AGENT_TOOLS_DIR knob; default ~/work.
FACADE = pathlib.Path(os.path.expanduser(os.environ.get("AGENT_TOOLS_DIR", "~/work"))) / "slack-facade"
OK, BAD = "✓", "✗"


def proc(pat):
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return bool(r.stdout.strip())


def main():
    # Same identity rule as app.py: the fleet name, which on some cloud VMs is NOT the
    # system hostname (that's the instance id, reset by cloud-init on every boot).
    name_file = HOME / ".claude/slack_facade/agent_name"
    host = (name_file.read_text().strip() if name_file.exists()
            else socket.gethostname())
    print(f"════════ {host} ════════")

    s = json.loads((HOME / ".claude/settings.json").read_text())
    hooks = s.get("hooks", {})
    matchers = {ev: [b.get("matcher", "(none)") for b in hooks.get(ev, [])]
                for ev in ("PreToolUse", "PostToolUse")}
    allow = (s.get("permissions") or {}).get("allow")

    WIDE = "Bash|Skill|Agent|Workflow|Write|Edit|NotebookEdit"
    for ev in ("PreToolUse", "PostToolUse"):
        good = matchers[ev] == [WIDE]
        print(f"  {OK if good else BAD} {ev:12} {matchers[ev]}")
    print(f"  {OK if hooks.get('SessionStart') else BAD} SessionStart hook present")
    print(f"    permissions.allow = {allow}  "
          f"({'blanket-allows -> no console prompts' if allow else 'none -> strict manual mode, WILL prompt'})")

    print("  ── paired units ──")
    for label, pat in (("body-watch", "body_bridge.py watch"),
                       ("consciousness", "python -u app.py")):
        up = proc(pat)
        print(f"  {OK if up else BAD} {label}")

    log = FACADE / "app.log"
    txt = log.read_text() if log.exists() else ""
    bot = re.findall(r"connected as bot user (\S+) \(bot_id (\S+)\)", txt)
    conn = "socket mode connected" in txt.split("connected as bot user")[-1] if bot else False
    print(f"  {OK if bot else BAD} slack face: bot_user={bot[-1][0] if bot else '?'} "
          f"bot_id={bot[-1][1] if bot else '?'} {'(socket connected)' if conn else ''}")

    app = (FACADE / "app.py").read_text()
    checks = [
        ("consciousness isolated (setting_sources=[])", "setting_sources=[]" in app),
        ("identity from hostname", "socket.gethostname()" in app),
        ("names the host in approvals", "'s console" in app),
        ("tools limited to delegate+check_status",
         "mcp__body__delegate_to_body" in app and "Bash" not in
         app.split("allowed_tools=[")[1].split("]")[0]),
    ]
    print("  ── consciousness safety ──")
    for label, good in checks:
        print(f"  {OK if good else BAD} {label}")

    guard = (HOME / ".claude/agent_bridge/guard_on").exists()
    await_f = HOME / ".claude/agent_bridge/awaiting_approval"
    print("  ── guard ──")
    print(f"    phone guard: {'ON' if guard else 'OFF'}")
    print(f"    awaiting_approval flag: "
          f"{await_f.read_text().strip() if await_f.exists() else '(none pending)'}")


if __name__ == "__main__":
    main()
