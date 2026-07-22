#!/usr/bin/env python3
"""PreToolUse hook ENTRY — a thin orchestrator over two clean layers:

  1. cage_policy   : the fundamental constraint (green vs gated). Transport-free.
  2. a transport   : how the owner is asked when something is gated. Swappable
                     (Telegram today; Slack DM / console later).

Flow: classify -> green? allow. gated? guard-off -> defer to Claude Code's own
prompt; else auto-allow-memory -> allow; else -> ask the active transport.

The 'guard' (only enforce approvals when guard_on exists) and the 'Always'-allow
memory are ORCHESTRATION/UX, not the cage — so they live here, not in cage_policy.

On any internal error it exits 0 (fail-open to the normal flow) so a bug here can
never freeze the session."""
import json, sys, time, pathlib, socket

sys.path.insert(0, "/home/astarostin/.claude/hooks")
import cage_policy
import transport_slack as transport   # swap this line to change approval channel

HOST = socket.gethostname()
LOG = pathlib.Path("/home/astarostin/.claude/agent_bridge/permission.log")
AWAITING = pathlib.Path("/home/astarostin/.claude/agent_bridge/awaiting_approval")
# The "cord": where gated approvals are asked — "slack" (default) or "console".
# Written by the `approvals` CLI (body) or the consciousness's set_approval_channel
# tool; both write the same file so either surface can flip it.
APPROVAL_CHANNEL = pathlib.Path("/home/astarostin/.claude/slack_facade/approval_channel")

def approval_channel():
    try:
        v = APPROVAL_CHANNEL.read_text().strip()
        return v if v in ("slack", "console") else "slack"
    except Exception:
        return "slack"
# Bases the owner tapped 'Always' for — an ad-hoc user allowlist (orchestration,
# not the fundamental cage). Read here; appended to on an 'always' decision.
AUTO_ALLOW = pathlib.Path("/home/astarostin/.claude/agent_bridge/auto_allow.txt")


def log(m):
    try:
        with LOG.open("a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n")
    except Exception:
        pass

def load_auto_allow():
    try:
        return set(w for w in AUTO_ALLOW.read_text().split() if w)
    except Exception:
        return set()

def auto_allowed_bash(command):
    """True only if EVERY segment's base command was previously 'Always'-allowed."""
    bases = cage_policy.base_tokens(command)
    if not bases:
        return False
    aa = load_auto_allow()
    return all(b in aa for b in bases)

def describe_tool(tool, ti):
    """One-screen summary of a non-Bash tool call, for the approval prompt."""
    def clip(s, n=400):
        s = str(s or "").strip()
        return s if len(s) <= n else s[:n] + " …"
    if tool == "Skill":
        args = clip(ti.get("args"), 300)
        return f"skill: {ti.get('skill', '?')}" + (f"\nargs: {args}" if args else "")
    if tool == "Agent":
        return (f"agent: {ti.get('subagent_type') or 'general-purpose'}"
                f"\ntask: {clip(ti.get('description'))}"
                f"\n\n{clip(ti.get('prompt'), 700)}")
    if tool == "Workflow":
        which = ti.get("name") or ti.get("scriptPath") or "(inline script)"
        return f"workflow: {clip(which)}"
    if tool in ("Write", "Edit", "NotebookEdit"):
        return f"{tool} → {ti.get('file_path') or ti.get('notebook_path') or '?'}"
    return clip(json.dumps(ti), 900)

def emit(decision, reason=""):
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision}}
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
    print(json.dumps(out))


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    AWAITING.unlink(missing_ok=True)   # any earlier approval is resolved by now
    transport.flush_context(payload.get("transcript_path"))

    tool = payload.get("tool_name") or ""
    ti = payload.get("tool_input") or {}

    verdict = cage_policy.classify(tool, ti)

    if verdict == "ungated":
        sys.exit(0)          # Write/Edit/… -> Claude Code's own layer decides
    if verdict == "green":
        emit("allow", "green zone (cage_policy)")
        sys.exit(0)

    # verdict == "gated" -> a human must decide. Everything below is transport/UX.
    if approval_channel() == "console":
        sys.exit(0)          # cord set to console -> defer to Claude Code's native prompt

    if tool == "Bash":
        cmd = ti.get("command", "") or ""
        if auto_allowed_bash(cmd):
            emit("allow", "auto-allowed (previously 'Always'-ed)")
            sys.exit(0)
        bases = cage_policy.base_tokens(cmd)
        base = bases[0] if bases else "command"
        shown = cmd if len(cmd) <= 1500 else cmd[:1500] + "\n...[truncated]"
    else:
        if tool in load_auto_allow():
            emit("allow", "auto-allowed (previously 'Always'-ed)")
            sys.exit(0)
        bases = [tool]
        base = tool
        shown = describe_tool(tool, ti)

    # If no transport is available (e.g. Telegram switched off), defer to the
    # native console prompt rather than blocking — the cage still holds.
    if not transport.available():
        sys.exit(0)

    header = f"🔒 Approve this {'command' if tool == 'Bash' else tool + ' call'} on {HOST}?"
    log(f"asking [{tool}]: {shown[:200]}")
    decision, comment = transport.ask(f"{header}\n\n{shown}", base)
    log(f"decision: {decision}" + (f" (note: {comment[:120]})" if comment else ""))

    if decision == "allow":
        transport.notify("✅ allowed")
        emit("allow", "approved via transport")
    elif decision == "always":
        try:
            existing = load_auto_allow()
            add = [b for b in bases if b not in existing]
            if add:
                with AUTO_ALLOW.open("a") as f:
                    for b in add:
                        f.write(b + "\n")
        except Exception:
            pass
        transport.notify(f"✅ allowed — I won't ask again for: {', '.join(bases) or base}"
                         f"  (send /allowlist to review · /allowlist clear to reset)")
        emit("allow", "approved + auto-allow via transport")
    elif decision == "deny":
        transport.notify("❌ denied")
        if comment:
            # Anatoly denied WITH a note ("no, archive them first"). A deny's reason is
            # shown to the body model, so hand his words back as an instruction — the
            # body then adapts rather than blindly retrying the identical command.
            reason = (f'Anatoly declined this and said: "{comment}". Treat that as his '
                      f"instruction — adjust course accordingly; do NOT just retry the "
                      f"same command.")
        else:
            reason = "denied via transport"
        emit("deny", reason)
    else:
        transport.notify("⏱ timed out — denying")
        # A timeout is NOT an active refusal — Anatoly just didn't answer in the window.
        # Anatoly's intent (2026-07-19): a short timeout should push the body toward a
        # cage-legal path, not a dead stop. So tell the model to look for an sbox workaround
        # before re-asking. (Reason text IS shown to the body model.)
        reason = (f"No reply within {transport.ASK_TIMEOUT}s, so this was auto-denied on "
                  f"TIMEOUT — not an active 'no' from Anatoly. Before re-requesting, look for "
                  f"a way to accomplish this INSIDE the sbox cage (writable ~/projects, /tmp, "
                  f"~/.cache; runs with no approval). Only re-ask the gated command if the task "
                  f"genuinely cannot be done in-cage.")
        emit("deny", reason)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"error: {e}")
        sys.exit(0)   # fail-open: never freeze the session on a hook bug
