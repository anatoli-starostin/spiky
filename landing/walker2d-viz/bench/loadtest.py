#!/usr/bin/env python3
"""Portable concurrency load test for the walker2d-viz demo server.

Opens N concurrent WebSocket *viewer* sessions against a running server, each selecting a REAL model and
consuming the streamed frames like a browser would, and measures the ACTUAL delivered frames-per-second per
session. Run it from anywhere (a laptop, nucstar, ...) against the deployed box — it does NOT need to run on
the server.

Only dependency: the `websockets` package (pip install websockets). Python 3.8+.

Examples
--------
  # sweep against the deployed box (its public wss endpoint), default model LUT-SAC c21:
  python loadtest.py --sessions 1,2,4,6,8,12,16,24

  # auto-ramp (1,2,4,8,16,...) and stop once per-session fps drops below target:
  python loadtest.py --max 32

  # a specific URL / model:
  python loadtest.py --url wss://89-169-96-79.sslip.io --model "Walker2d SAC baseline" --sessions 4,8,16

  # a local/dev server over plain ws:
  python loadtest.py --url ws://127.0.0.1:8765 --sessions 1,2,4

The default target is $WALKER2D_WS or wss://89-169-96-79.sslip.io (the deployment box). Both ws:// and wss://
are supported. Each session picks `--model` (a real, inference-running actor) so the measured rate reflects a
real viewer load, not a no-op placeholder.
"""
import argparse
import asyncio
import json
import os
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("This needs the 'websockets' package:  pip install websockets")

DEFAULT_URL = os.environ.get("WALKER2D_WS") or "wss://89-169-96-79.sslip.io"
DEFAULT_MODEL = "fastlut_lse (exp19)"   # the server's own DEFAULT_ACTOR; "Walker2d LUT-SAC c21"
                                        # was the old default but was pruned from the demo 2026-08-15


async def one_session(url, model, warmup, window, out, idx):
    """One viewer: connect, select `model`, count 'state' frames over the measurement window.

    out[idx] = ("fps", value) | ("full", message) | ("err", detail)
    """
    frames = 0
    try:
        async with websockets.connect(url, open_timeout=15, max_size=None) as ws:
            picked = False
            t_start = time.time() + warmup
            t_end = t_start + window
            while time.time() < t_end:
                try:
                    d = json.loads(await asyncio.wait_for(ws.recv(), 5))
                except asyncio.TimeoutError:
                    break
                t = d.get("type")
                if t == "server_full":                      # graceful capacity refusal
                    out[idx] = ("full", d.get("message", "")); return
                if t == "actors" and not picked:
                    # The server silently ignores an unknown actor name (set_actor no-ops when the name
                    # isn't in its registry), so without this check a typo or a pruned actor would leave
                    # the session measuring whatever the DEFAULT actor is -- a different CPU cost, quietly
                    # reported as if it were the requested one. Fail the session instead.
                    available = d.get("actors") or []
                    if available and model not in available:
                        out[idx] = ("err", f"actor {model!r} not offered by server; have: {available}")
                        return
                    await ws.send(json.dumps({"cmd": "actor", "name": model})); picked = True
                elif t == "state" and time.time() >= t_start:
                    frames += 1
    except Exception as e:
        out[idx] = ("err", repr(e)[:90]); return
    out[idx] = ("fps", frames / window)


async def run_level(url, model, n, sps, warmup, window):
    out = {}
    tasks = []
    for i in range(n):
        tasks.append(asyncio.create_task(one_session(url, model, warmup, window, out, i)))
        await asyncio.sleep(0.02)                           # tiny stagger so connects don't all land at once
    await asyncio.gather(*tasks)
    fps = [v for kind, v in out.values() if kind == "fps"]
    full = sum(1 for kind, _ in out.values() if kind == "full")
    err = sum(1 for kind, _ in out.values() if kind == "err")
    accepted = len(fps)
    mean = sum(fps) / len(fps) if fps else 0.0
    mn = min(fps) if fps else 0.0
    meeting = sum(1 for f in fps if f >= sps * 0.9)         # within 10% of target
    return dict(n=n, accepted=accepted, full=full, err=err, mean=mean, min=mn,
                frac=(meeting / accepted if accepted else 0.0))


def ramp(mx):
    out, n = [], 1
    while n <= mx:
        out.append(n)
        n = n * 2 if n < 8 else n + 8
    return out


async def main():
    ap = argparse.ArgumentParser(description="walker2d-viz concurrency load test (client-side).")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"server WS URL, ws:// or wss:// (default: {DEFAULT_URL})")
    ap.add_argument("--host", help="convenience: build ws://HOST:PORT (overrides --url when given)")
    ap.add_argument("--port", type=int, default=8765, help="port for --host (default 8765)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"actor selected per session (default: {DEFAULT_MODEL!r})")
    ap.add_argument("--sessions", help="comma list of N to sweep, e.g. 1,2,4,6,8,12,16,24")
    ap.add_argument("--max", type=int, help="auto-ramp 1,2,4,8,16,... up to N; stops once min fps < 0.85*target")
    ap.add_argument("--sps", type=float, default=30.0, help="target frames/s per session (server default 30)")
    ap.add_argument("--warmup", type=float, default=2.0, help="seconds discarded before measuring (default 2)")
    ap.add_argument("--window", type=float, default=8.0, help="measurement window seconds (default 8)")
    a = ap.parse_args()

    url = f"ws://{a.host}:{a.port}" if a.host else a.url
    if a.sessions:
        levels = [int(x) for x in a.sessions.split(",") if x.strip()]
    elif a.max:
        levels = ramp(a.max)
    else:
        levels = [1, 2, 4, 6, 8, 12, 16, 24]

    print(f"target {url}  model {a.model!r}  target={a.sps:g} fps/session  window={a.window:g}s")
    print(f"{'N':>4} {'accept':>6} {'refused':>7} {'err':>4} {'mean fps':>9} {'min fps':>8} {'>=90%tgt':>9}  status")
    for n in levels:
        r = await run_level(url, a.model, n, a.sps, a.warmup, a.window)
        if r["full"] and r["accepted"] == 0:
            status = "ALL CAPPED"
        elif r["full"]:
            status = "PARTIAL CAP"
        elif r["accepted"] < n:
            status = "conn-fail"
        elif r["min"] >= a.sps * 0.85:
            status = "ok"
        else:
            status = "DEGRADING"
        print(f"{n:>4} {r['accepted']:>6} {r['full']:>7} {r['err']:>4} "
              f"{r['mean']:>9.1f} {r['min']:>8.1f} {100 * r['frac']:>8.0f}%  {status}", flush=True)
        if a.max and r["accepted"] == n and r["min"] and r["min"] < a.sps * 0.85:
            print("(min per-session fps dropped below 85% of target — stopping ramp)")
            break


if __name__ == "__main__":
    asyncio.run(main())
