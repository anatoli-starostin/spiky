---
name: Telegram bridge
description: Two-way Telegram bridge for the active Claude Code session — script paths, bot/chat identifiers, hook wiring, and HTML formatting trick.
type: reference
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
# Telegram bridge for Claude Code

The user (Anatoli, Telegram username `astarostin`) wants to monitor and chat with the
active Claude Code session from his phone. Implemented as a pair of scripts plus a
Stop-hook — the bridge is **session-local**, not a sidecar `claude -p` worker.

## Bot
- Token: stored inline in the scripts (rotate by editing in two places).
- Chat ID: `<REDACTED_TELEGRAM_CHAT_ID>` (private chat with Anatoli).

## Files
- `~/.claude/telegram_bridge/tg_monitor.py` — long-poll `getUpdates`, prints
  `TG_MSG: <text>` lines on stdout for each new user message. Tracks offset in
  `monitor_offset.txt`. Run under Claude Code's `Monitor` tool with
  `persistent=true`; each line becomes a task-notification.
- `~/.claude/telegram_bridge/tg_send.sh` — small `curl` wrapper for ad-hoc sends
  from inside the session: `tg_send.sh "text"`.
- `~/.claude/hooks/telegram_notify.sh` → execs `telegram_notify.py` — Stop-hook
  that reads the transcript JSONL, extracts the **last assistant text block** of
  the turn, and forwards it to Telegram. So my normal terminal replies are mirrored
  to Telegram automatically; I rarely need `tg_send.sh` directly.
- `~/.claude/telegram_bridge/bridge.py` — the *standalone* `claude -p`-spawning
  bridge (Option C from the design discussion). Built but **not used** because
  Anatoli wanted the Telegram conversation to share context with the active
  terminal session.

## Hook configuration
`~/.claude/settings.json` has a `Stop` hook entry pointing at
`telegram_notify.sh`. Don't double-register.

## Approval UI
Permission prompts from `telegram_permission.py` send an **inline keyboard**
(✅ Approve / ❌ Deny buttons) alongside the prompt text. Button taps generate
`callback_query` updates; the hook acknowledges them via `answerCallbackQuery`
(closes the loading spinner) and resolves allow/deny. **Text replies still
work** as a fallback: `yes / y / go / ok / allow / approve` → allow;
`no / n / stop / deny / block` → deny. Anything else triggers a re-prompt
with the keyboard.

## Formatting tricks
- Telegram by default treats `**bold**` as literal asterisks. The hook converts
  Markdown → Telegram **HTML** (`<b>`, `<i>`, `<code>`, `<pre>`) and sends with
  `parse_mode=HTML`. HTML-escapes `<>&` first to prevent injection. Falls back
  to plain text on API rejection.
- Avoid MarkdownV2 — too many escape requirements (`. - ( ) +` etc).
- Telegram message limit: 4096 chars. The hook truncates to 4000 to leave room
  for HTML tags.

## How to start the inbound monitor
```
Monitor(
  description="Telegram inbound messages",
  persistent=true,
  command="/home/starost/spiky/.venv/bin/python -u /home/starost/.claude/telegram_bridge/tg_monitor.py",
  timeout_ms=3600000,
)
```

## Limitations
- Per-session. New `claude` invocation needs the Monitor restarted.
- Inbound chat_id is hardcoded; only Anatoli's account is honored.
- Stop hook fires after every turn — including ones with no text content
  (tool-only turns send nothing, by design in the script).
