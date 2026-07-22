#!/usr/bin/env python3
"""Slack-DM approval TRANSPORT (body side of the two-Claude relay).

The body's hook can't talk to Slack. Instead it drops an approval REQUEST in a
shared dir and blocks polling for a DECISION. The consciousness (app.py), which is
the one thing wired into Slack, watches for requests, DMs Anatoly, and writes back
the decision. Only the consciousness touches Slack; the body stays a plain session.

File protocol in ~/.claude/slack_facade/approvals/ :
  req_<id>.json  {id, prompt, base, host, ts, timeout}   written by body
  seen_<id>      (empty)   the consciousness claims a request
  dec_<id>.json  {decision: allow|deny|always|timeout}   written by consciousness

available() lets permission_gate fall back to the console prompt if the
consciousness isn't alive, so the cage never blocks on a dead face.
"""
import json, os, socket, time, pathlib

BASE = pathlib.Path.home() / ".claude" / "slack_facade"
APPROVALS = BASE / "approvals"
ALIVE = BASE / "consciousness_alive"        # consciousness heartbeat (unix ts)
# Reuse the existing awaiting flag so the Slack face's check_status can report WHERE
# the approval is waiting; value now names the channel ("slack").
AWAITING = pathlib.Path.home() / ".claude" / "agent_bridge" / "awaiting_approval"
# Short on purpose: a gated command that gets no Slack reply within this window is
# auto-denied on timeout, which nudges the body to find a cage-legal (sbox) workaround
# rather than idling on a human. Anatoly's call (2026-07-19): 300s -> 60s.
ASK_TIMEOUT = 60


def available():
    """True iff the consciousness heartbeat is fresh (< 60s)."""
    try:
        return (time.time() - float(ALIVE.read_text().strip())) < 60
    except Exception:
        return False


def _rid():
    # unique without Math.random: pid + millisecond clock
    return f"{os.getpid()}_{int(time.time() * 1000)}"


def ask(prompt, base="command", timeout=ASK_TIMEOUT):
    """Post an approval request for the consciousness and block for the decision.
    Returns (decision, comment): decision is 'allow'|'deny'|'always'|'timeout';
    comment is an optional note Anatoly typed alongside a denial (e.g. "no, archive
    them first") to hand back to the body as his instruction. Empty otherwise."""
    APPROVALS.mkdir(parents=True, exist_ok=True)
    rid = _rid()
    req = APPROVALS / f"req_{rid}.json"
    dec = APPROVALS / f"dec_{rid}.json"
    req.write_text(json.dumps({
        "id": rid, "prompt": prompt, "base": base,
        "host": socket.gethostname(), "ts": int(time.time()), "timeout": timeout,
    }))
    AWAITING.write_text("slack")
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(1.5)
            if dec.exists():
                try:
                    d = json.loads(dec.read_text())
                    return d.get("decision", "deny"), (d.get("comment") or "")
                except Exception:
                    return "deny", ""
        return "timeout", ""
    finally:
        for f in (req, dec, APPROVALS / f"seen_{rid}"):
            try:
                f.unlink()
            except Exception:
                pass
        AWAITING.unlink(missing_ok=True)


def notify(text):
    """No-op: for the Slack channel the consciousness posts the outcome (✅/❌) in
    the DM itself, since it's the one holding the conversation."""
    return


def flush_context(transcript_path):
    """No-op for Slack. (Telegram used this to push pending assistant text to the
    phone right before the prompt; the Slack DM relay doesn't need it.)"""
    return
