"""exp_c29 — self-updating Slack bar for the 4-arm x 3-seed constants sweep (#75).

TWO WAVES on one bar: wave 1 draws its anchors with lutorch's `balanced` policy, wave 2
with `canonical_full_coverage`, and nothing else differs between them. They share this
message rather than getting one each, because the interesting reading is the two side by
side -- a per-arm difference that survives both samplers is a different claim from one
that only appears under one of them.

Twenty-four runs would drown a chat message if listed in full, so the layout is: a header
with done/running/queued, the RUNNING cells in detail, and a per-arm roll-up whose cells
switch from the live MJX proxy to the 100-episode CPU reference as soon as that lands.
Roughly twenty lines, and it answers "where is it" and "what is it saying" at a glance.

TWO NUMBERS, NEVER CONFLATED. During training a cell shows the MJX proxy in
parentheses -- 20 episodes, horizon 1000, perturbation-free MJX physics. That is a
watching number, not a result, and this chapter has a specific reason to distrust it:
c21 read 425 at iter 1500 and finished at 5287. The number that decides the experiment
is the deterministic 100-episode CPU reference, and a cell only shows it bare (no
parentheses) once eval_const_cpu.py has written it.

Exits once BOTH sentinels exist (SWEEP_DONE, SWEEP_DONE_CANONICAL), or when the wall
clock passes --max-hours. Deliberately not `pgrep -f`: exp_c25's refresher matched its
own command line and kept posting a frozen number for nine hours after the work had
finished.

--handle reuses an EXISTING bar instead of creating one. That is what keeps wave 2 on the
same Slack message: this process can be restarted with new rendering code and the reader
sees the same message update in place, rather than a second bar appearing halfway
through and the first one dying frozen.

Usage:
  python slack_bar.py --task <BODY_TASK id> [--handle <existing>] [--interval 150]
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+([\d.]+)%\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
FAIL = re.compile(r"Traceback|CUDA out of memory|Killed|MemoryError")

ARMS = [("none", "17-dim baseline (magnitude-blind)"),
        ("grid", "16 tiled thresholds")]
# `random` and `clumped` were dropped mid-flight on 2026-08-02. Two wave-1 runs had
# already been launched by then and were left to finish; they are reported separately
# rather than deleted, because compute already spent is worth measuring even for an arm
# that is no longer part of the design.
SALVAGE = [("c29", "random", 0), ("c29", "clumped", 0)]
SEEDS = (0, 1, 2)
# (tag, log prefix, sentinel, label) — a 2x2: two capacity partitions x two anchor
# samplers. Waves 1/3 draw with `balanced`, waves 2/4 with `canonical_full_coverage`;
# waves 1/2 are nap6/tph64 and waves 3/4 nap5/tph128. Every cell holds the same 49,152
# learnable params, so the grid isolates partition and sampler from capacity. Wave 4 was
# added once wave 2 showed the sampler alone moves a fixed-seed cell by up to 919 --
# a swing larger than the none-vs-grid effect, which makes the sampler a factor to
# measure rather than a detail to fix.
WAVES = [("c29", "cell", "SWEEP_DONE", "wave 1 · balanced · nap6/tph64"),
         ("c29c", "cellc", "SWEEP_DONE_CANONICAL",
          "wave 2 · canonical_full_coverage · nap6/tph64"),
         ("c29m", "cellm", "SWEEP_DONE_MIDCAP",
          "wave 3 · balanced · nap5/tph128 (param-matched, 640 comparators)"),
         ("c29mc", "cellmc", "SWEEP_DONE_MIDCAP_CANONICAL",
          "wave 4 · canonical_full_coverage · nap5/tph128")]
PER_WAVE = len(ARMS) * len(SEEDS)
TOTAL = PER_WAVE * len(WAVES)
REF = ("_anchors x hard nap6/tph64. exp_c13 3-seed mean for this cell: 4346 +/- 546 "
       "(non-deterministic). exp_c18 hyperplane nap6/tph32, 6 seeds, deterministic: "
       "4308 +/- 500._")


def bar(pct, width=12):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def read(tag, prefix, arm, seed):
    st = dict(state="queued", pct=0.0, cpu=None)
    ev = os.path.join(HERE, f"lut_sac_{tag}_{arm}_s{seed}_cpueval.json")
    if os.path.exists(ev):
        try:
            st["cpu"] = json.load(open(ev))["cpu_reference_mean"]
        except (OSError, ValueError, KeyError):
            pass
    try:
        txt = open(os.path.join(HERE, f"{prefix}_{arm}_s{seed}.log"),
                   errors="replace").read()
    except OSError:
        return st
    rows = ITER.findall(txt)
    d = DONE.search(txt)
    if d:
        st.update(state="done", pct=100.0, best=float(d.group(1)),
                  cov=float(rows[-1][3]) if rows else 0.0)
        return st
    # Failure is checked BEFORE progress: a run that printed 9,000 iterations and then
    # died must not keep showing as a healthy 90%.
    if FAIL.search(txt):
        return dict(st, state="failed", pct=0.0)
    if rows:
        it, tot, ret, cov, best = rows[-1]
        return dict(st, state="running", pct=100.0 * int(it) / int(tot), it=int(it),
                    tot=int(tot), best=float(best), cov=float(cov))
    return dict(st, state="starting", pct=0.0)


def cell(v):
    if v["cpu"] is not None:
        return f"{v['cpu']:.0f}"
    if v["state"] == "done":
        return f"({v['best']:.0f})"
    if v["state"] == "failed":
        return "✗"
    if v["state"] == "running":
        return f"({v['best']:.0f})…"
    return "·" if v["state"] == "queued" else "…"


def render():
    st = {(w, a, s): read(tag, pre, a, s)
          for w, (tag, pre, _, _) in enumerate(WAVES) for a, _ in ARMS for s in SEEDS}
    n_done = sum(1 for v in st.values() if v["state"] == "done")
    n_fail = sum(1 for v in st.values() if v["state"] == "failed")
    running = [(k, v) for k, v in st.items() if v["state"] in ("running", "starting")]
    n_cpu = sum(1 for v in st.values() if v["cpu"] is not None)
    n_queued = TOTAL - n_done - n_fail - len(running)
    pct = sum(v["pct"] for v in st.values()) / TOTAL

    lines = [f"*{n_done}/{TOTAL} trained* · {len(running)} running · "
             f"{n_queued} queued · {n_cpu}/{TOTAL} CPU-evaluated"
             + (f" · *{n_fail} FAILED*" if n_fail else "")]
    if running:
        lines.append("")
        for (w, arm, seed), v in sorted(running):
            lab = f"{'w2 ' if w else ''}{arm} s{seed}"
            if v["state"] == "starting":
                lines.append(f"`{lab:<16}` {bar(0)}  starting…")
            else:
                lines.append(f"`{lab:<16}` {bar(v['pct'])} {v['pct']:3.0f}% · "
                             f"{v['it']:,}/{v['tot']:,} · MJX best {v['best']:.0f} "
                             f"· row-cov {v['cov']:.0f}%")
    for w, (_, _, _, label) in enumerate(WAVES):
        lines.append("")
        lines.append(f"*{label}*")
        for arm, desc in ARMS:
            cells = " · ".join(f"s{s} {cell(st[(w, arm, s)])}" for s in SEEDS)
            lines.append(f"`{arm:<8}` {cells}   _{desc}_")
    sal = [(arm, seed, read(tag, "cell", arm, seed)) for tag, arm, seed in SALVAGE]
    sal = [(a, s, v) for a, s, v in sal if v["state"] != "queued" or v["cpu"] is not None]
    if sal:
        lines.append("")
        lines.append("_dropped arms, wave-1 runs already launched and left to finish:_ "
                     + " · ".join(f"`{a} s{s}` {cell(v)}" for a, s, v in sal))
    lines.append("")
    lines.append("_bare number = 100-ep deterministic CPU reference (the result); "
                 "(parenthesised) = live 20-ep MJX proxy, not comparable._")
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None,
                    help="reuse an existing bar (keeps the SAME Slack message) instead "
                         "of posting a new one")
    ap.add_argument("--interval", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=16.0,
                    help="hard stop, so a stuck sweep cannot leave this posting forever")
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c29 — do fixed constants cure anchor magnitude blindness? (#75)",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c29.handle"), "w").write(h)
    print(f"bar {h} ({'reused' if a.handle else 'new'})", flush=True)

    t0 = time.time()
    sentinels = [os.path.join(HERE, s) for _, _, s, _ in WAVES]
    while True:
        pct, n_done, n_fail, n_cpu, body = render()
        progress.progress_update(h, pct=pct, stats=REF + "\n" + body)
        finished = all(os.path.exists(s) for s in sentinels)
        timeout = (time.time() - t0) > a.max_hours * 3600
        if finished or timeout:
            _, n_done, n_fail, n_cpu, body = render()
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=(n_fail == 0 and not timeout),
                final_text=(f"exp_c29 sweep {'finished' if finished else 'TIMED OUT'} "
                            f"{stamp} — {n_done}/{TOTAL} trained, "
                            f"{n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
