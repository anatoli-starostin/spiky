#!/usr/bin/env python3
"""Slack progress bars — a shared brick for tracking slow work in a thread.

A reusable primitive so any agent / experiment on any fleet host can post a
progress bar into a Slack thread and update it *in place* as work advances
(training steps, long downloads, multi-stage jobs). It rides the same rails as
the rest of the façade: the body-side writes small record files; the face
(app.py, which holds the bot token) posts once with `chat_postMessage` and then
edits that same message with `chat_update` — exactly the post-once / update-many
pattern the approval buttons already use.

── Why a file rendezvous (and why ~/.cache) ─────────────────────────────────
The caller (a training loop) usually runs INSIDE the sbox cage, which is
read-write only in ~/projects, /tmp and ~/.cache and has no network. The face
runs OUTSIDE the cage and holds the Slack connection. So the two meet at a
GREEN-ZONE directory both can reach: `~/.cache/slack_facade/progress/`. The
caller writes there with zero approvals and zero network; the face reads there
and does the Slack I/O. No cage_policy change is needed, and progress updates
never cost the owner an approval tap.

── Files per bar ────────────────────────────────────────────────────────────
  <handle>.json        written by the CALLER (pct / stats / state / rev)
  <handle>.sent.json   written ONLY by the face reaper (message ts, posted rev)
Two single-writer files, so the caller and the reaper never race on one file.

── Body-side API (import this in an experiment; pure stdlib, cage-safe) ──────
    import progress
    h = progress.progress_start("exp042 pretrain", task=TASK_ID)   # -> handle
    ...
    progress.progress_update(h, step=8000, total=16000,
                             stats="val_bpb 1.243 · eta ~12m")
    ...
    progress.progress_done(h, ok=True, final_text="val_bpb 1.201")

  `task` binds the bar to a delegated BODY_TASK so it posts in that task's
  thread; alternatively pass explicit `channel=`/`thread_ts=`. Style defaults to
  emoji squares (🟩⬜); pass style="unicode" for a `██░░` block bar.

── CLI (for bash / non-Python loops; also cage-safe) ────────────────────────
    h=$(python3 progress.py start --task "$TASK_ID" --label "exp042")
    python3 progress.py update "$h" --step 8000 --total 16000 --stats "eta ~12m"
    python3 progress.py done "$h" --text "val_bpb 1.201"      # add --fail on failure

── Face-side (app.py) ───────────────────────────────────────────────────────
    import progress
    asyncio.create_task(progress.reaper(web, log, task_loader=body_bridge.load))
  The reaper polls every few seconds and only edits when the record changed, so
  many rapid updates coalesce into one edit — a light guard well under Slack's
  ~1 update/sec/channel limit (validation cadence is far slower anyway).
"""
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Green-zone rendezvous: writable from inside the sbox cage, readable by the
# out-of-cage face. NOT under ~/.claude (that is read-only in the cage).
PROGRESS_DIR = Path.home() / ".cache" / "slack_facade" / "progress"

_STYLES = ("emoji", "unicode")


# =============================================================================
# Rendering (pure; safe to unit-test without Slack)
# =============================================================================

def render_bar(pct, style: str = "emoji", width: int = 10) -> str:
    """A width-cell bar for a 0..100 percentage.

    emoji   -> 🟩×filled ⬜×empty (renders identically desktop/mobile, no
               monospace needed).
    unicode -> `█×filled ░×empty` wrapped in backticks so the fill stays
               width-stable in Slack's proportional font.
    """
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        pct = 0.0
    pct = max(0.0, min(100.0, pct))
    width = max(1, int(width))
    filled = int(round(pct / 100.0 * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    if style == "unicode":
        return "`" + "█" * filled + "░" * empty + "`"
    return "🟩" * filled + "⬜" * empty


def render_message(rec: dict) -> str:
    """The full Slack message for a record: a heading line + the bar + stats."""
    state = rec.get("state", "active")
    pct = rec.get("pct", 0)
    if state == "done":
        pct = 100
    label = rec.get("label") or "progress"
    stats = (rec.get("stats") or "").strip()
    final_text = (rec.get("final_text") or "").strip()
    bar = render_bar(pct, rec.get("style", "emoji"), rec.get("width", 10))
    if state == "done":
        head, tail = f"✅ *{label}* — done", (final_text or stats)
    elif state == "failed":
        head, tail = f"❌ *{label}* — failed", (final_text or stats)
    else:
        head, tail = f"*{label}*", stats
    line2 = f"{bar} {int(round(pct))}%" + (f" · {tail}" if tail else "")
    return f"{head}\n{line2}"


# =============================================================================
# Record I/O (atomic single-writer helpers)
# =============================================================================

def _rec_path(handle: str) -> Path:
    return PROGRESS_DIR / f"{handle}.json"


def _sent_path(handle: str) -> Path:
    return PROGRESS_DIR / f"{handle}.sent.json"


def _atomic_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)   # atomic on POSIX: a reader never sees a half-file


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# =============================================================================
# Body-side API (called by the experiment / agent; stdlib only, cage-safe)
# =============================================================================

def progress_start(label: str, *, task: str = None, channel: str = None,
                   thread_ts: str = None, style: str = "emoji", width: int = 10,
                   pct: float = 0.0, stats: str = "") -> str:
    """Create a bar and return its handle. Bind it to a thread by either `task`
    (a BODY_TASK id — posts in that task's thread) or explicit
    `channel`+`thread_ts`. Writes one green-zone record; no network, no approval."""
    if style not in _STYLES:
        style = "emoji"
    handle = uuid.uuid4().hex[:8]
    now = time.time()
    _atomic_write(_rec_path(handle), {
        "handle": handle, "task": task, "channel": channel, "thread_ts": thread_ts,
        "label": label, "style": style, "width": int(width),
        "pct": float(pct), "stats": stats, "state": "active", "final_text": "",
        "rev": 0, "created": now, "updated": now,
    })
    return handle


def _bump(handle: str, **changes) -> None:
    rec = _read(_rec_path(handle))
    if not rec:
        return
    rec.update(changes)
    rec["rev"] = int(rec.get("rev", 0)) + 1
    rec["updated"] = time.time()
    _atomic_write(_rec_path(handle), rec)


def progress_update(handle: str, pct: float = None, stats: str = None, *,
                    step: int = None, total: int = None) -> None:
    """Advance a bar. Give either `pct` (0..100) or `step`+`total` (pct is
    derived). `stats` is the free-text suffix (e.g. "val_bpb 1.243 · eta ~12m").
    Cheap: just rewrites the record — the face coalesces edits."""
    changes = {}
    if step is not None and total:
        pct = 100.0 * float(step) / float(total)
    if pct is not None:
        changes["pct"] = max(0.0, min(100.0, float(pct)))
    if stats is not None:
        changes["stats"] = stats
    if changes:
        _bump(handle, **changes)


def progress_done(handle: str, ok: bool = True, final_text: str = "") -> None:
    """Terminal edit: flip the bar to ✅ done (fills to 100%) or ❌ failed.
    ALWAYS call this (even on failure) so a dead poller can't leave a stale bar."""
    _bump(handle, state=("done" if ok else "failed"),
          final_text=final_text, **({"pct": 100.0} if ok else {}))


# =============================================================================
# Face-side reaper (called by app.py; async, holds the Slack client)
# =============================================================================

def _resolve_target(rec: dict, task_loader):
    """(channel, thread_ts) for a record: explicit fields win, else inherit from
    the bound BODY_TASK via task_loader. (None, None) if not resolvable yet."""
    if rec.get("channel"):
        return rec["channel"], rec.get("thread_ts")
    tid = rec.get("task")
    if tid and task_loader:
        t = task_loader(tid)
        if t:
            return t.get("channel"), t.get("post_thread")
    return None, None


async def _reap_one(web, rec: dict, task_loader, log) -> None:
    handle = rec.get("handle")
    if not handle:
        return
    channel, thread_ts = _resolve_target(rec, task_loader)
    if not channel:
        return  # target not known yet (task record not written / no channel) — retry next tick
    sent = _read(_sent_path(handle)) or {}
    msg = render_message(rec)

    if not sent.get("ts"):
        r = await web.chat_postMessage(channel=channel, thread_ts=thread_ts, text=msg)
        if r.get("ok"):
            _atomic_write(_sent_path(handle), {
                "ts": r["ts"], "channel": channel,
                "posted_rev": rec.get("rev", 0), "posted_state": rec.get("state"),
                "finalized": rec.get("state") in ("done", "failed"),
            })
            if log:
                log.info("progress %s posted to %s (%s)", handle, channel, thread_ts)
        return

    if sent.get("finalized"):
        return
    # Only edit when something actually changed — coalesces bursts of updates
    # into one chat_update (the light throttle; also well under the rate limit).
    if rec.get("rev") == sent.get("posted_rev") and rec.get("state") == sent.get("posted_state"):
        return
    await web.chat_update(channel=sent["channel"], ts=sent["ts"], text=msg)
    sent["posted_rev"] = rec.get("rev", 0)
    sent["posted_state"] = rec.get("state")
    sent["finalized"] = rec.get("state") in ("done", "failed")
    _atomic_write(_sent_path(handle), sent)


async def reaper(web, log=None, task_loader=None, poll: float = 2.5,
                 ttl: float = 86400.0) -> None:
    """Poll the green-zone progress dir; post new bars and edit changed ones.

    `web` is the AsyncWebClient; `task_loader` resolves a task id -> task dict
    (pass body_bridge.load). Robust by construction: one bad record can never
    crash the loop or the face. Terminal bars older than `ttl` are reaped."""
    import asyncio
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        await asyncio.sleep(poll)
        try:
            recs = list(PROGRESS_DIR.glob("*.json"))
        except Exception:
            continue
        for f in recs:
            if f.name.endswith(".sent.json"):
                continue
            rec = _read(f)
            if not rec:
                continue
            try:
                await _reap_one(web, rec, task_loader, log)
            except Exception:
                if log:
                    log.exception("progress reaper failed for %s", rec.get("handle"))
            # Janitor: drop terminal bars long after they finished.
            try:
                if (rec.get("state") in ("done", "failed")
                        and time.time() - rec.get("updated", 0) > ttl):
                    f.unlink(missing_ok=True)
                    _sent_path(rec["handle"]).unlink(missing_ok=True)
            except Exception:
                pass


# =============================================================================
# CLI (for bash / non-Python callers; writes the same green-zone records)
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(prog="progress.py", description="Slack progress bars")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start")
    s.add_argument("--label", required=True)
    s.add_argument("--task"); s.add_argument("--channel"); s.add_argument("--thread-ts", dest="thread_ts")
    s.add_argument("--style", default="emoji", choices=_STYLES)
    s.add_argument("--width", type=int, default=10)
    s.add_argument("--pct", type=float, default=0.0); s.add_argument("--stats", default="")

    s = sub.add_parser("update")
    s.add_argument("handle")
    s.add_argument("--pct", type=float); s.add_argument("--stats")
    s.add_argument("--step", type=int); s.add_argument("--total", type=int)

    s = sub.add_parser("done")
    s.add_argument("handle")
    s.add_argument("--fail", action="store_true"); s.add_argument("--text", default="")

    a = p.parse_args()
    if a.cmd == "start":
        print(progress_start(a.label, task=a.task, channel=a.channel, thread_ts=a.thread_ts,
                             style=a.style, width=a.width, pct=a.pct, stats=a.stats))
    elif a.cmd == "update":
        progress_update(a.handle, pct=a.pct, stats=a.stats, step=a.step, total=a.total)
    elif a.cmd == "done":
        progress_done(a.handle, ok=not a.fail, final_text=a.text)


if __name__ == "__main__":
    main()
