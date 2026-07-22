#!/usr/bin/env python3
"""Task queue shared by two halves of the agent:

  * the Slack *consciousness* (app.py) -- writes tasks via delegate_to_body,
    reads results via check_status, and (a reaper loop) relays finished results
    back into the Slack thread in its own voice;
  * the *body* -- THIS host's Claude Code CLI session, which is told about new
    tasks by a harness Monitor running `body_bridge.py watch`, executes them
    with its real tools (approvals ride the out-of-band channel — Slack DM / console), and
    reports the outcome back with `body_bridge.py done|error <id>`.

Plain stdlib, no deps: importable by app.py and runnable as a CLI by the body.
"""
import json
import sys
import time
import uuid
from pathlib import Path

TASKS = Path.home() / ".claude" / "slack_facade" / "tasks"
TASKS.mkdir(parents=True, exist_ok=True)

# Written by the permission hook for exactly as long as a human is being asked
# to approve something. Lets check_status distinguish "waiting on Anatoly" from
# "merely slow" -- without it the consciousness could only guess (and once guessed
# right, then talked itself out of it because the task still said 'pending').
AWAITING = Path.home() / ".claude" / "agent_bridge" / "awaiting_approval"


def awaiting_where():
    """Where the pending approval is waiting: 'console' (a prompt in this machine's
    Claude session -- attachable via tmux) or 'slack' (the owner's DM). None if nothing pending."""
    try:
        return AWAITING.read_text().strip() or "console"
    except Exception:
        return None


def effective_status(t: dict) -> str:
    """The task's status, upgraded to awaiting-approval if a human is being asked
    right now and this task is the one in flight."""
    if t.get("status") in ("pending", "running") and AWAITING.exists():
        return "awaiting-approval"
    return t.get("status", "unknown")


def _f(tid: str) -> Path:
    return TASKS / f"{tid}.json"


def create(instruction: str, channel: str, post_thread, thread: str) -> str:
    tid = uuid.uuid4().hex[:8]
    save({
        "id": tid, "instruction": instruction,
        "channel": channel, "post_thread": post_thread, "thread": thread,
        "status": "pending", "result": None,
        "announced": False, "posted": False, "created": time.time(),
    })
    return tid


def load(tid: str):
    p = _f(tid)
    return json.loads(p.read_text()) if p.exists() else None


def save(t: dict) -> None:
    _f(t["id"]).write_text(json.dumps(t))


def set_status(tid: str, status: str, result=None) -> None:
    t = load(tid)
    if not t:
        return
    t["status"] = status
    if result is not None:
        t["result"] = result
    save(t)


def all_tasks():
    for p in sorted(TASKS.glob("*.json")):
        try:
            yield json.loads(p.read_text())
        except Exception:
            pass


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "watch":
        # Emit each new pending task once, for the harness Monitor to surface to
        # the body. Mark announced BEFORE printing so a restart can't double-fire.
        while True:
            for t in all_tasks():
                if t["status"] == "pending" and not t.get("announced"):
                    t["announced"] = True
                    save(t)
                    me = Path(__file__).resolve()
                    print(f"BODY_TASK {t['id']} :: {t['instruction'].strip()}", flush=True)
                    # Hand out the EXACT report form. It must be this absolute-path
                    # shape to match the permission hook's auto-allow pattern --
                    # otherwise filing the result costs Anatoly a needless approval tap.
                    print(f"  ↳ report back with EXACTLY this form (pre-approved, no "
                          f"prompt): python3 {me} done {t['id']} <<'EOF' … EOF   "
                          f"(result on stdin; use 'error' instead of 'done' if it failed)",
                          flush=True)
                    # The body works UNATTENDED: Anatoly is NOT at this console and
                    # cannot see anything you print or any prompt you raise here. So do
                    # NOT ask him anything at the console and do NOT block waiting on
                    # input -- never use AskUserQuestion, never pause for a permission
                    # you can't get, never sit on a plan waiting for approval. If you
                    # need a DECISION, a clarification, or you're declining something
                    # risky: don't stall -- write your question/options as the RESULT
                    # via the 'done' form above (phrased for Anatoly, e.g. "Install
                    # lean texlive (~1.5GB) or full (~5GB)? I'll proceed on your reply").
                    # It reaches him in Slack and he sends a follow-up task with the
                    # answer. Reporting-back is how you reach him; the console is a
                    # dead end. (Interactive tools are fine ONLY if he has actually
                    # attached to this session and is talking to you live.)
                    print(f"  ↳ UNATTENDED: never AskUserQuestion / never block for input "
                          f"— a decision or clarification is itself a valid RESULT: report "
                          f"it back with the done form and Anatoly answers in Slack.",
                          flush=True)
                    print(f"  ↳ To SEND A PICTURE (a plot, chart, screenshot): save it to a "
                          f"file, then put a marker [[IMAGE:/absolute/path.png]] anywhere in "
                          f"your done result (one per image). It gets uploaded into the Slack "
                          f"thread automatically — do NOT try to send files yourself.",
                          flush=True)
            time.sleep(2)
    elif cmd in ("running", "done", "error") and len(sys.argv) > 2:
        result = sys.stdin.read().strip() if cmd in ("done", "error") else None
        set_status(sys.argv[2], cmd, result)
        print(f"task {sys.argv[2]} -> {cmd}")
    elif cmd == "show" and len(sys.argv) > 2:
        print(json.dumps(load(sys.argv[2]), indent=2))
    elif cmd == "list":
        for t in all_tasks():
            print(f"{t['id']}  {t['status']:16}  {t['instruction'][:60]}")
    else:
        print("usage: body_bridge.py watch | running|done|error <id> | show <id> | list")


if __name__ == "__main__":
    main()
