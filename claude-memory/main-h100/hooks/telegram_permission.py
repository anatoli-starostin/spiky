#!/usr/bin/env python3
"""PreToolUse hook: route would-be permission prompts to Telegram.

Logic:
  - Read JSON payload from stdin (tool_name, tool_input, ...).
  - For Read/Grep/Glob/Edit/Write/NotebookEdit/etc.: do nothing (exit 0)
    so Claude Code's normal permission flow runs.
  - For Bash: classify command. Safe-prefix matches → exit 0 (let normal
    flow handle it). Otherwise: ask Telegram (yes/no), block up to 120s,
    return a permissionDecision allow/deny.
  - Pauses the tg_monitor poller via /tmp/tg_monitor_paused so messages
    aren't stolen.
"""
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

TOKEN = "<REDACTED_TELEGRAM_BOT_TOKEN>"
CHAT_ID = "<REDACTED_TELEGRAM_CHAT_ID>"
PAUSE_FLAG = pathlib.Path("/tmp/tg_monitor_paused")
LOG_FILE = pathlib.Path("/home/starost/.claude/telegram_bridge/permission.log")
ASK_TIMEOUT_SECONDS = 120

API = f"https://api.telegram.org/bot{TOKEN}"

# Bash commands that match any of these prefixes don't trigger a Telegram ask.
# Each entry is matched against every '|', '&&', '||', ';' segment of the command.
SAFE_BASH_PATTERNS = [
    r"^\s*ls(\s|$)",
    r"^\s*cat\s",
    r"^\s*head(\s|$)",
    r"^\s*tail(\s|$)",
    r"^\s*grep\s",
    r"^\s*egrep\s",
    r"^\s*rg\s",
    r"^\s*find\s",
    r"^\s*ps(\s|$)",
    r"^\s*pgrep(\s|$)",
    r"^\s*nvidia-smi",
    r"^\s*git\s+(status|diff|log|show|branch|remote|reflog|stash\s+list)",
    r"^\s*which\s",
    r"^\s*echo\s",
    r"^\s*pwd(\s|$)",
    r"^\s*awk\s",
    r"^\s*sed\s+-n",
    r"^\s*wc(\s|$)",
    r"^\s*sort(\s|$)",
    r"^\s*uniq(\s|$)",
    r"^\s*jq\s",
    r"^\s*column\s",
    r"^\s*paste\s",
    r"^\s*cut\s",
    r"^\s*tr\s",
    r"^\s*basename\s",
    r"^\s*dirname\s",
    r"^\s*realpath\s",
    r"^\s*date(\s|$)",
    r"^\s*hostname(\s|$)",
    r"^\s*whoami(\s|$)",
    r"^\s*mkdir\s+-p",
    r"^\s*chmod\s+\+x\s",
    r"^\s*kill\s+-0\s",
    r"^\s*until\s",  # poll loops
    r"^\s*\[\s+",     # test brackets
]

ALWAYS_SAFE_TOOLS = {
    "Read", "Grep", "Glob", "Edit", "Write", "NotebookEdit",
    "TodoWrite", "Skill", "Agent", "Task", "TaskCreate", "TaskGet",
    "TaskList", "TaskUpdate", "TaskOutput", "TaskStop",
    "ToolSearch", "Monitor", "ScheduleWakeup", "AskUserQuestion",
    "ExitPlanMode", "EnterPlanMode", "EnterWorktree", "ExitWorktree",
    "WebFetch", "WebSearch", "PushNotification",
    "CronCreate", "CronDelete", "CronList", "RemoteTrigger",
}


def log(msg):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def is_safe_bash(command: str) -> bool:
    """Check whether all top-level commands in the shell line match a safe prefix.

    Uses shlex tokenisation so `|`, `;`, `&&`, `||` characters inside quoted
    arguments (e.g. inside a grep regex like 'foo|bar') don't cause spurious
    sub-command splits.
    """
    cmd = command.strip()
    if not cmd:
        return True
    import shlex
    try:
        tokens = shlex.split(cmd, comments=False, posix=True)
    except ValueError:
        return False
    if not tokens:
        return True
    SEPS = {"|", "&&", "||", ";"}
    parts = []
    cur = []
    for t in tokens:
        if t in SEPS:
            if cur:
                parts.append(" ".join(cur))
                cur = []
        else:
            cur.append(t)
    if cur:
        parts.append(" ".join(cur))
    return all(any(re.match(p, part) for p in SAFE_BASH_PATTERNS) for part in parts)


def http_post(url, data, timeout=10):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def http_get(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def send_message(text, reply_markup=None):
    try:
        body = {"chat_id": CHAT_ID, "text": text}
        if reply_markup is not None:
            body["reply_markup"] = json.dumps(reply_markup)
        http_post(f"{API}/sendMessage", body, timeout=10)
    except Exception:
        pass


APPROVE_KEYBOARD = {
    "inline_keyboard": [[
        {"text": "✅ Approve", "callback_data": "allow"},
        {"text": "❌ Deny", "callback_data": "deny"},
    ]]
}


def answer_callback(callback_query_id, text=""):
    try:
        body = {"callback_query_id": callback_query_id}
        if text:
            body["text"] = text
        http_post(f"{API}/answerCallbackQuery", body, timeout=5)
    except Exception:
        pass


def ask_telegram(prompt: str, timeout_seconds: int = ASK_TIMEOUT_SECONDS) -> str:
    """Block-poll Telegram for approval via inline-keyboard buttons (tap
    ✅ Approve / ❌ Deny) or a text reply ('yes' / 'no' / etc).
    Returns 'allow', 'deny', or 'timeout'."""
    PAUSE_FLAG.touch()
    try:
        try:
            data = http_get(f"{API}/getUpdates?timeout=1", timeout=5)
            results = data.get("result", [])
            offset = (results[-1]["update_id"] + 1) if results else 0
        except Exception:
            offset = 0
        send_message(prompt, reply_markup=APPROVE_KEYBOARD)
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            remaining = max(1, int(deadline - time.time()))
            poll = min(remaining, 20)
            try:
                data = http_get(f"{API}/getUpdates?timeout={poll}&offset={offset}", timeout=poll + 5)
            except Exception:
                continue
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1

                # Inline-button tap → callback_query
                cb = upd.get("callback_query")
                if cb:
                    if str(cb.get("from", {}).get("id", "")) != CHAT_ID:
                        continue
                    answer_callback(cb.get("id", ""))  # dismiss the spinner
                    cb_data = (cb.get("data", "") or "").strip().lower()
                    if cb_data == "allow":
                        return "allow"
                    if cb_data == "deny":
                        return "deny"
                    continue

                # Text reply fallback
                msg = upd.get("message") or upd.get("edited_message") or {}
                if str(msg.get("chat", {}).get("id", "")) != CHAT_ID:
                    continue
                text = (msg.get("text", "") or "").strip().lower()
                if text in ("y", "yes", "go", "ok", "allow", "approve"):
                    return "allow"
                if text in ("n", "no", "stop", "deny", "block"):
                    return "deny"
                send_message("Tap a button or reply 'yes' / 'no'.", reply_markup=APPROVE_KEYBOARD)
        return "timeout"
    finally:
        PAUSE_FLAG.unlink(missing_ok=True)


def emit_decision(decision: str, reason: str = ""):
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
        }
    }
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
    print(json.dumps(out))


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        log(f"failed to parse stdin: {e}")
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    if tool_name in ALWAYS_SAFE_TOOLS:
        emit_decision("allow", "always-safe tool auto-allow")
        sys.exit(0)

    if tool_name == "Bash":
        cmd = tool_input.get("command", "") or ""
        if is_safe_bash(cmd):
            emit_decision("allow", "safe-prefix auto-allow")
            sys.exit(0)
        truncated = cmd if len(cmd) <= 1500 else cmd[:1500] + "\n...[truncated]"
        prompt = f"🔒 Approve Bash?\n\n{truncated}\n\nReply: yes / no"
        log(f"asking Bash: {cmd[:200]}")
        decision = ask_telegram(prompt)
        log(f"decision: {decision}")
        if decision == "allow":
            send_message("✅ allowed")
            emit_decision("allow", "approved via Telegram")
        elif decision == "deny":
            send_message("❌ denied")
            emit_decision("deny", "user denied via Telegram")
        else:
            send_message("⏱ timed out — denying")
            emit_decision("deny", "Telegram approval timed out")
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"unexpected error: {e}")
        sys.exit(0)
