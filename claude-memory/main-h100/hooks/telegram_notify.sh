#!/usr/bin/env bash
# Stop-hook: forwards the last assistant text message of the turn to Telegram.
exec /home/starost/spiky/.venv/bin/python /home/starost/.claude/hooks/telegram_notify.py
