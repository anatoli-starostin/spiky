"""Slack progress bar for exp18: 2 arms x 3 seeds = 6 runs, arms run sequentially.

Shows the arm currently in flight plus its per-seed progress, and the learned taus as
each seed lands. Bound to BODY_TASK a4e53bde.

Cage-safe: `progress` writes only under ~/.cache/slack_facade/progress. No network.

Usage:  python progress_monitor_exp18.py [existing_handle]
"""
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress                                                   # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TASK = "a4e53bde"
LABEL = "exp18 · exponential critic vs plain-sum LUT critic · 2 arms × 3 seeds"
ARMS = [("exp", "exp18_lut-lse-lutcrit-t32"),
        ("plain", "exp18ctl_lut-lse-plaincrit-t32")]
SEEDS = [0, 1, 2]
TOTAL = 768
POLL = 20
MAX_H = 3.0
PAT = re.compile(r"\[upd\s+([\d,]+)/([\d,]+)\].*?ep_ret\s+(-?[\d.]+)")


def d_of(folder):
    return os.path.join(BASE, folder)


def parse_last(folder, s):
    f = os.path.join(d_of(folder), f"ppo_s{s}.log")
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


def done_run(folder, s):
    return os.path.exists(os.path.join(d_of(folder), f"ppo_s{s}.json"))


def finished(folder, s):
    """(final_return, tau_actor, tau_critic) from a completed run."""
    try:
        h = json.load(open(os.path.join(d_of(folder), f"ppo_s{s}.json")))["history"]
        return h[-1]["ep_ret_mean"], h[-1].get("tau_actor"), h[-1].get("tau_critic")
    except Exception:
        return None, None, None


def main():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        h = sys.argv[1].strip()
        print("adopted handle", h, flush=True)
    else:
        h = progress.progress_start(LABEL, task=TASK, width=12, stats="starting…")
        print("handle", h, flush=True)

    t0 = time.time()
    frac0 = None
    n_runs = len(ARMS) * len(SEEDS)
    while time.time() - t0 < MAX_H * 3600:
        frac = 0.0
        segs = []
        for tag, folder in ARMS:
            parts = []
            for s in SEEDS:
                if done_run(folder, s):
                    frac += 1.0
                    parts.append(f"s{s}✅")
                else:
                    info = parse_last(folder, s)
                    if info:
                        frac += info["upd"] / TOTAL
                        parts.append(f"s{s} {info['upd']}/{TOTAL} r{info['ret']:.0f}")
                    else:
                        parts.append(f"s{s}·")
            segs.append(f"[{tag}] " + " ".join(parts))
        pct = 100.0 * frac / n_runs
        if frac0 is None:
            frac0 = frac
        el = time.time() - t0
        gained = frac - frac0
        eta = ""
        if gained > 0.01 and el > 0 and pct < 100:
            eta = f" · eta ~{(n_runs - frac) / (gained / el) / 60:.0f}m"
        progress.progress_update(h, pct=pct, stats=" · ".join(segs) + eta)
        if all(done_run(f, s) for _, f in ARMS for s in SEEDS):
            break
        time.sleep(POLL)

    ok = all(done_run(f, s) for _, f in ARMS for s in SEEDS)
    if ok:
        bits = []
        for tag, folder in ARMS:
            res = [finished(folder, s) for s in SEEDS]
            m = statistics.mean(r[0] for r in res)
            ta = statistics.mean(r[1] for r in res if r[1] is not None)
            tc = [r[2] for r in res if r[2] is not None and r[2] == r[2]]
            bits.append(f"{tag} {m:.0f} (τa {ta:.4f}"
                        + (f", τc {statistics.mean(tc):.4f}" if tc else "")
                        + ")")
        progress.progress_done(
            h, ok=True,
            final_text="6/6 · " + " · ".join(bits)
                       + " · vs exp17 5404 / exp13 2359 / exp10 5488")
    else:
        n = sum(done_run(f, s) for _, f in ARMS for s in SEEDS)
        progress.progress_done(h, ok=False, final_text=f"only {n}/{n_runs} runs finished")
    print("finalized", h, "ok=", ok, flush=True)


if __name__ == "__main__":
    main()
