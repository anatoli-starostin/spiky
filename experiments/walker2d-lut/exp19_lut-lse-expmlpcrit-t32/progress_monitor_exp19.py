"""Slack progress bar for exp19: exponential-readout MLP critic, 3 seeds in parallel.

Surfaces both taus live — tau_actor (does it still drift UP toward the sum, as in exp17?)
and tau_critic (does the critic's own readout go toward max?), which is the question this
experiment exists to answer.

Bound to BODY_TASK 828921db. Cage-safe: `progress` writes only under
~/.cache/slack_facade/progress. No network.

Usage:  python progress_monitor_exp19.py [existing_handle]
"""
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = "828921db"
LABEL = "exp19 · exponential MLP-critic readout (τ_c) + LSE actor · 3 seeds"
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


def taus(s):
    try:
        h = json.load(open(os.path.join(HERE, f"ppo_s{s}.json")))["history"]
        return h[-1].get("tau_actor"), h[-1].get("tau_critic")
    except Exception:
        return None, None


def main():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        h = sys.argv[1].strip()
        print("adopted handle", h, flush=True)
    else:
        h = progress.progress_start(LABEL, task=TASK, width=12, stats="starting…")
        print("handle", h, flush=True)

    t0 = time.time()
    frac0 = None
    while time.time() - t0 < MAX_H * 3600:
        frac, seg = 0.0, []
        for s in SEEDS:
            if done_run(s):
                frac += 1.0
                ta, tc = taus(s)
                seg.append(f"s{s}✅"
                           + (f" τa{ta:.4f} τc{tc:.4f}" if ta is not None else ""))
            else:
                info = parse_last(s)
                if info:
                    frac += info["upd"] / TOTAL
                    seg.append(f"s{s} {info['upd']}/{TOTAL} r{info['ret']:.0f}")
                else:
                    seg.append(f"s{s}…")
        pct = 100.0 * frac / len(SEEDS)
        if frac0 is None:
            frac0 = frac
        el = time.time() - t0
        gained = frac - frac0
        eta = ""
        if gained > 0.01 and el > 0 and pct < 100:
            eta = f" · eta ~{(len(SEEDS) - frac) / (gained / el) / 60:.0f}m"
        progress.progress_update(h, pct=pct, stats=" · ".join(seg) + eta)
        if all(done_run(s) for s in SEEDS):
            break
        time.sleep(POLL)

    ok = all(done_run(s) for s in SEEDS)
    if ok:
        fs = [json.load(open(os.path.join(HERE, f"ppo_s{s}.json")))["final_ep_ret"]
              for s in SEEDS]
        tt = [taus(s) for s in SEEDS]
        ta = statistics.mean(t[0] for t in tt)
        tc = statistics.mean(t[1] for t in tt)
        progress.progress_done(
            h, ok=True,
            final_text=(f"3/3 · final {statistics.mean(fs):.0f} ("
                        + ", ".join(f"{v:.0f}" for v in fs)
                        + f") vs exp17 5404 / exp10 5488 · τ_actor {ta:.4f} "
                          f"(exp17 0.0887, init 0.05) · τ_critic {tc:.4f} (init 0.25)"))
    else:
        n = sum(done_run(s) for s in SEEDS)
        progress.progress_done(h, ok=False, final_text=f"only {n}/3 seeds produced a result")
    print("finalized", h, "ok=", ok, flush=True)


if __name__ == "__main__":
    main()
