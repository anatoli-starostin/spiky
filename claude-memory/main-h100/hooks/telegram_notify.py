#!/usr/bin/env python3
"""Stop-hook: read the Claude Code transcript, find the last assistant text
message of the turn, and forward it to Telegram. Sends nothing if the final
assistant entry has no text content (e.g. tool-only turn).
"""
import html
import json
import re
import sys
import urllib.parse
import urllib.request


def md_to_html(text):
    """Minimal Markdown -> Telegram HTML conversion.
    Supports: **bold**, *italic*, `code`, ```code blocks```.
    Escapes <, >, & first so user content cannot inject HTML.
    """
    code_blocks = []
    def stash_block(m):
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks)-1}\x00"
    text = re.sub(r"```(?:\w*\n)?(.*?)```", stash_block, text, flags=re.DOTALL)

    inline_codes = []
    def stash_inline(m):
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes)-1}\x00"
    text = re.sub(r"`([^`\n]+)`", stash_inline, text)

    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)

    def restore_inline(m):
        return f"<code>{html.escape(inline_codes[int(m.group(1))])}</code>"
    text = re.sub(r"\x00IC(\d+)\x00", restore_inline, text)

    def restore_block(m):
        return f"<pre>{html.escape(code_blocks[int(m.group(1))])}</pre>"
    text = re.sub(r"\x00CB(\d+)\x00", restore_block, text)
    return text

TOKEN = "<REDACTED_TELEGRAM_BOT_TOKEN>"
CHAT_ID = "<REDACTED_TELEGRAM_CHAT_ID>"


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        return

    last_text = ""
    try:
        with open(transcript_path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") != "assistant":
                    continue
                msg = ev.get("message", {})
                buf = []
                for c in msg.get("content", []):
                    if c.get("type") == "text":
                        t = c.get("text", "")
                        if t:
                            buf.append(t)
                if buf:
                    last_text = "\n".join(buf)
    except Exception:
        return

    text = last_text.strip()
    if not text:
        return
    text = text[:4000]
    text_html = md_to_html(text)
    try:
        body = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text": text_html,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data=body,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        # fallback to plain text without parse_mode
        try:
            body = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data=body,
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass


if __name__ == "__main__":
    main()
