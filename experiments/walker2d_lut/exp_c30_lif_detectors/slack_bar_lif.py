"""exp_c30 — self-updating Slack bar for the 3-seed LIF-detector SAC sweep (#75).

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS, not a network call. This
process writes `~/.cache/slack_facade/progress/<handle>.json` -- a green-zone directory
that is read-write inside the sbox cage -- and the face (`app.py`, outside the cage, which
holds the bot token) reaps it and does the Slack I/O. So the bar runs under `sbox` and
costs no approval.

TWO NUMBERS, NEVER CONFLATED. During training a cell shows the MJX proxy in parentheses --
20 episodes, horizon 1000, perturbation-free MJX physics. That is a watching number, and
this chapter distrusts it for a specific reason: c21 read 425 at iter 1500 and finished at
5287. The number that decides the experiment is the deterministic 100-episode CPU
reference, and a cell only shows it bare once eval_lif_cpu.py has written it.

The proxy reads LOW early here for an extra reason that is not a fault: training anneals
eps 2.0 -> 0.3 while every eval is pinned at eps 0.3, so early evals score a policy in a
regime it has not been trained into yet.

Exits once SWEEP_DONE_LIF exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and kept posting a frozen number for nine
hours after the work had finished.

Usage:
  python slack_bar_lif.py --task <BODY_TASK id> [--handle <existing>] [--interval 150]
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
                  r"\s+\|\s+row-cov\s+([\d.]+)%\s+\|\s+eps\s+([\d.]+)\s+\|\s+tbit\s+"
                  r"([\d.]+)\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
FAIL = re.compile(r"Traceback|CUDA out of memory|Killed|MemoryError|XlaRuntimeError")

SEEDS = (0, 1, 2)
TOTAL = len(SEEDS)
SENTINEL = "SWEEP_DONE_LIF"
REF = ("_LIF-detector actor, nap6/tph32, 87,361 params — NOT param-matched to the "
       "49,152-param LUT actors. Anchor: exp_c18 hyperplane at the same shape, "
       "6 seeds, deterministic: 4308 +/- 500._")


def read(seed):
    """(state, pct, text) for one cell."""
    log = os.path.join(HERE, f"cell_s{seed}.log")
    cpu = os.path.join(HERE, f"lif_sac_c30_s{seed}_cpueval.json")
    if os.path.exists(cpu):
        d = json.load(open(cpu))
        return "cpu", 100.0, (f"*{d['cpu_reference_mean']:.0f}* "
                              f"±{d['cpu_reference_std']:.0f} "
                              f"({d['full_length']}/100 full)")
    if not os.path.exists(log):
        return "queued", 0.0, "queued"
    txt = open(log, errors="replace").read()
    # Failure is checked BEFORE progress: a run that printed 9,000 iterations and then
    # died is a failure, not a 90%-complete cell.
    if FAIL.search(txt):
        return "fail", 0.0, "FAILED — see log"
    dn = DONE.search(txt)
    if dn:
        return "done", 100.0, f"trained, best MJX {float(dn.group(1)):.0f} · awaiting CPU eval"
    ms = ITER.findall(txt)
    if not ms:
        return "run", 0.0, "starting"
    it, tot, ret, cov, eps, tbit, best = ms[-1]
    pct = 100.0 * int(it) / max(1, int(tot))
    return "run", pct, (f"{int(it):,}/{int(tot):,} · ({float(ret):.0f}) · "
                        f"rows {float(cov):.0f}% · eps {float(eps):.2f} · "
                        f"tbit {float(tbit):.2f}")


def render():
    cells = {s: read(s) for s in SEEDS}
    n_cpu = sum(1 for s in SEEDS if cells[s][0] == "cpu")
    n_done = sum(1 for s in SEEDS if cells[s][0] in ("done", "cpu"))
    n_fail = sum(1 for s in SEEDS if cells[s][0] == "fail")
    n_run = sum(1 for s in SEEDS if cells[s][0] == "run")
    pct = sum(c[1] for c in cells.values()) / TOTAL

    lines = [f"*{n_done}/{TOTAL} trained* · {n_run} running · "
             f"{TOTAL - n_done - n_fail - n_run} queued · {n_cpu}/{TOTAL} CPU-evaluated",
             ""]
    for s in SEEDS:
        state, _, text = cells[s]
        mark = {"cpu": "✅", "done": "🧮", "run": "⏳", "fail": "❌"}.get(state, "•")
        lines.append(f"{mark} `seed {s}` {text}")
    if n_cpu:
        vals = [json.load(open(os.path.join(
            HERE, f"lif_sac_c30_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        lines += ["", f"*mean over {len(vals)} seeds: {m:.0f}" +
                  (f" ± {sd:.0f}*" if len(vals) > 1 else "*") +
                  f"  ·  exp_c18 hyperplane anchor: 4308 ± 500"]
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy, not comparable._"]
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=16.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c30 — LIF-detector actor on Walker2d SAC, 3 seeds (#75)",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c30.handle"), "w").write(h)
    print(f"bar {h} ({'reused' if a.handle else 'new'})", flush=True)

    t0 = time.time()
    sentinel = os.path.join(HERE, SENTINEL)
    while True:
        pct, n_done, n_fail, n_cpu, body = render()
        progress.progress_update(h, pct=pct, stats=REF + "\n" + body)
        finished = os.path.exists(sentinel)
        timeout = (time.time() - t0) > a.max_hours * 3600
        if finished or timeout:
            _, n_done, n_fail, n_cpu, body = render()
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=(n_fail == 0 and not timeout),
                final_text=(f"exp_c30 LIF-detector sweep "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
