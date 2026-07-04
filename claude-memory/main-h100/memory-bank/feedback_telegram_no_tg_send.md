---
name: feedback-telegram-no-tg-send
description: "Don't call tg_send.sh — the Stop hook already auto-forwards every assistant text reply to Telegram. Calling tg_send.sh duplicates the message and triggers a separate permission prompt each time."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Don't call `tg_send.sh` — the Stop hook mirrors replies automatically

**Rule**: Reply normally in the terminal. Do not call `~/.claude/telegram_bridge/tg_send.sh`.

**Why**: `~/.claude/hooks/telegram_notify.sh` is wired as a `Stop` hook in `~/.claude/settings.json`. After each turn it reads the transcript JSONL, extracts my last assistant text block, and forwards it to Telegram. So any normal terminal reply is already mirrored. Calling `tg_send.sh` on top of that:
- Duplicates the message (user sees two copies)
- Triggers a permission prompt via `telegram_permission.py` inline keyboard (extra friction)
- Adds approval-timeout delays when the user is afk

**How to apply**:
- For Telegram-driven sessions (inbound `tg_monitor.py` running), just reply normally. The hook handles outbound.
- Exception: if the user genuinely needs a notification BEFORE my turn ends (e.g. partway through a long tool-use chain), `tg_send.sh "..."` is OK — but this should be rare.
- Don't try to send Telegram-specific HTML formatting via `tg_send.sh`. The Stop hook converts Markdown → HTML automatically and reuses my normal reply.
