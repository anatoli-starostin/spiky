"""Slack progress bar for the gpustar exp10 reproduction: 1 group x 3 seeds, run in
parallel. Adapted from src/progress_monitor12.py (the monitor nebius used for bench12,
which is exp10's provenance) — same parsing and the same green-zone `progress` rendezvous,
narrowed to a single tph=32 group and pointed at this folder's logs.

Read-only w.r.t. the training run. Cage-safe: `progress` writes only under
~/.cache/slack_facade/progress, so this costs no approval and touches no network.

The bar is bound to BODY_TASK 5ec7983d — the task that owns the reproduction run — so it
posts in that run's Slack thread.

Usage:  python progress_monitor_repro.py [existing_handle]

Passing an existing handle ADOPTS that bar instead of creating a new one, so the monitor
can be restarted without posting a duplicate message into the thread.
"""
import os
import re
import statistics
import sys
import time

sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = "5ec7983d"
SEEDS = [0, 1, 2]
TOTAL = 768
POLL = 20
MAX_H = 3.0
PAT = re.compile(r"\[upd\s+([\d,]+)/([\d,]+)\].*?ep_ret\s+(-?[\d.]+)")


def parse_last(s):
    f = os.path.join(HERE, f"ppo_s{s}.log")
    if not os.path.exists(f):
        return None
    last = None
    try:
        for line in open(f, errors="replace"):
            m = PAT.search(line)
            if m:
                last = m
    except OSError:
        return None
    return None if last is None else dict(upd=int(last.group(1).replace(",", "")),
                                          ret=float(last.group(3)))


def done_run(s):
    return os.path.exists(os.path.join(HERE, f"ppo_s{s}.json"))


def main():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        h = sys.argv[1].strip()          # adopt an existing bar (restart without duplicating)
        print("adopted handle", h, flush=True)
    else:
        h = progress.progress_start("exp10 reproduction on gpustar · fastlut tph32 · 3 seeds",
                                    task=TASK, width=12, stats="starting…")
        print("handle", h, flush=True)
    t0 = time.time()
    frac0 = None                          # progress already banked when THIS monitor started
    while time.time() - t0 < MAX_H * 3600:
        frac, seg = 0.0, []
        for s in SEEDS:
            if done_run(s):
                frac += 1.0
                seg.append(f"s{s}✅")
            else:
                info = parse_last(s)
                if info:
                    frac += info["upd"] / TOTAL
                    seg.append(f"s{s} {info['upd']}/{TOTAL} r{info['ret']:.0f}")
                else:
                    seg.append(f"s{s}…")
        pct = 100.0 * frac / len(SEEDS)
        # ETA from progress made SINCE this monitor started — not total frac / uptime,
        # which reads ~0m when the monitor is attached to an already-running job.
        if frac0 is None:
            frac0 = frac
        el = time.time() - t0
        eta = ""
        gained = frac - frac0
        if gained > 0.01 and el > 0 and pct < 100:
            rem = (len(SEEDS) - frac) / (gained / el)
            eta = f" · eta ~{rem / 60:.0f}m"
        progress.progress_update(h, pct=pct, stats=" · ".join(seg) + eta)
        if all(done_run(s) for s in SEEDS):
            break
        time.sleep(POLL)

    ok = all(done_run(s) for s in SEEDS)
    if ok:
        import json
        fs = [json.load(open(os.path.join(HERE, f"ppo_s{s}.json")))["final_ep_ret"]
              for s in SEEDS]
        txt = (f"3/3 seeds · final {statistics.mean(fs):.0f} "
               f"(" + ", ".join(f"{v:.0f}" for v in fs) + ") · reference 5488")
        progress.progress_done(h, ok=True, final_text=txt)
    else:
        n = sum(done_run(s) for s in SEEDS)
        progress.progress_done(h, ok=False, final_text=f"only {n}/3 seeds produced a result")
    print("finalized", h, "ok=", ok, flush=True)


if __name__ == "__main__":
    main()
