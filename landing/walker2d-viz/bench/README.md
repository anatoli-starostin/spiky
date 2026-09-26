# walker2d-viz load test

`loadtest.py` — a client-side concurrency load generator for the demo server. It opens N concurrent
WebSocket viewer sessions, each selecting a **real** model (default `fastlut_lse (exp19)`, the server's own
`DEFAULT_ACTOR`, which runs actual inference every step), consumes the streamed frames like a browser, and reports the **delivered
frames-per-second per session**. Point it at the deployed box or any server; it does not run on the server.

## Requirements
- Python 3.8+
- `pip install websockets`  (the only dependency)

## Usage
```sh
# sweep against the deployed box (its public wss endpoint), default model:
python loadtest.py --sessions 1,2,4,6,8,12,16,24

# auto-ramp 1,2,4,8,16,... and stop once per-session fps drops below target:
python loadtest.py --max 64

# a specific URL / model:
python loadtest.py --url wss://89-169-96-79.sslip.io --model "Spiking LUT quantised (handcrafted SNN)" --sessions 8,16,32

# a local/dev server over plain ws:
python loadtest.py --url ws://127.0.0.1:8765 --sessions 1,2,4
```

Flags: `--url` (ws:// or wss://, default `$WALKER2D_WS` or `wss://89-169-96-79.sslip.io`), `--host/--port`
(convenience for `ws://HOST:PORT`), `--model`, `--sessions A,B,C` or `--max N`, `--sps` (target, default 30),
`--warmup`/`--window` (default 2 s / 8 s).

## If the actor name is wrong

The server ignores an unknown actor name rather than erroring, so a typo — or a model that has since been
pruned from the demo — would otherwise leave every session measuring the *default* actor and reporting its
frame rate under the name you asked for. `loadtest.py` therefore checks the requested name against the
actors the server advertises and fails the session instead, listing what is on offer.

## Output
One row per N: sessions **accepted** vs **refused** (server_full), mean & min per-session fps, the fraction
meeting ≥90% of target, and a status (`ok` / `DEGRADING` / `PARTIAL CAP` / `ALL CAPPED`). Degradation begins
where **min fps** falls meaningfully below the target sps.

To find the *true* server ceiling (not just the configured `MAX_SESSIONS`), temporarily raise the server's
`MAX_SESSIONS` (set it in the box's `.env` and `docker compose up -d`), run the sweep, then restore it —
otherwise the cap refuses extra sessions with `server_full` before the box is actually saturated.

There is also a local, 2-vCPU-emulated benchmark (`taskset -c 0,1`) kept alongside the deploy notes for a
hardware-controlled number.
