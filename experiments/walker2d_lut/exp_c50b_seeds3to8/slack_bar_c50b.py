"""exp_c50b — self-updating Slack bar for exp_c50's six additional seeds (3-8).

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network
call, so this runs under `sbox` and costs no approval.

THIS IS NOT A NEW CONFIGURATION. It is exp_c50 continued: the same module, the same
clamp-with-no-floor, the same everything, on seeds 3-8. c50's first three seeds settled the
MECHANISM (the learned delay distribution matched c36's seed for seed) but left the RETURN
unresolved at |t| 0.71. These six bring it to n=9 pooled.

WHAT THE BAR REPORTS. Two means, kept separate and both shown: the six new seeds on their
own, and the pooled n=9 including c50's original 0/1/2 (4447.2 / 3719.6 / 1156.3). The
pooled number is the one that answers the question; the six-seed number is what tells you
whether the new seeds behave like the old ones.

TWO NUMBERS, NEVER CONFLATED. During training a cell shows the MJX proxy in parentheses --
20 episodes, horizon 1000, perturbation-free physics. That is a watching number; c21 read
425 at iteration 1,500 and finished at 5,287. The number that decides the experiment is the
deterministic 100-episode CPU reference, shown bare once eval_mhl_cpu.py has written it.

Exits once SWEEP_DONE_C50B exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and posted a frozen number for nine hours.

Usage:
  python slack_bar_c50b.py --task <BODY_TASK id> [--handle <existing>] [--interval 150]
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
C50 = os.path.join(HERE, "..", "exp_c50_no_delay_clamp")
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+cell-cov\s+([\d.]+)%\s+\|\s+digit\s+([\d.]+)\s+\|\s+"
                  r"nosp\s+([\d.]+)\s+\|\s+eff\s+([\d.]+)/(\d+)\s+\|\s+"
                  r"Tbkt\s+([\d.]+)\s+\|\s+Tcr\s+([\d.]+)\s+\|\s+best\s+([-\d.]+)\s+\|"
                  r"\s+([\d.]+)m", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
FAIL = re.compile(r"Traceback|CUDA out of memory|Killed|MemoryError|XlaRuntimeError")

SEEDS = (3, 4, 5, 6, 7, 8)
TOTAL = len(SEEDS)
PRIOR = {0: 4447.2, 1: 3719.6, 2: 1156.3}       # exp_c50 seeds 0/1/2, already banked
TAKEOFF = 3000.0
SENTINEL = "SWEEP_DONE_C50B"
REF = ("_*exp_c50b — six MORE seeds (3–8) of exp_c50.* Not a new configuration: the same "
       "unified `LIFMultiHeadLUT`, 1 head × *128 tables × 1 LIF detector × 16 buckets*, "
       "per-table betas, stock 0.1 table init, `delay_init_std=0`, SORT_FORM='rank', "
       "temperatures trainable, 31,360 params, and the same delay clamp with its LOWER "
       "bound removed (`clamp(delay, −inf, t_window)`; the upper cap is kept for float32 "
       "safety in the reference's cumsum membrane). Parity *105/105*.\n"
       "_*Why more seeds and not a further bisect.* c50 seeds 0/1/2 settled the MECHANISM: "
       "the learned delay distribution matched c36's seed for seed — means 0.533/0.464/"
       "0.505 vs 0.542/0.460/0.612, negative fractions 37.4/41.5/41.1% vs 37.7/40.8/38.9%, "
       "0.00% left on the retained cap — and that is a measurement over 6,528 parameters, "
       "not a 3-sample mean. What it did NOT settle is the RETURN: *3108 ± 1729* (2/3) vs "
       "c49's *2233 ± 1259* is |t| 0.71, and vs c36's *4246 ± 298* is |t| 1.12, down from "
       "c49's |t| 2.69._\n"
       "_*The c42b lesson applied.* A configuration that fails about half the time has a "
       "≥6% chance of showing 3/3 and a matching chance of 1/3, so a takeoff rate at n=3 "
       "is not a measurement. Seeds 3–8 bring exp_c50 to *n=9 pooled*, the standard c42b "
       "established. Note seed 2 was c49's BEST seed (3174) and c50's WORST (1156) — that "
       "is a takeoff lottery, not a seed-level property._\n"
       "_*What to look for.* Pooled n=9 near *4246* → the clamp was the whole c36 gap. "
       "Pooled materially below → a residual remains and the bisect resumes (membrane "
       "formulation, then bucket-digit path, then soft partition). The takeoff COUNT out "
       "of 9 is the number that matters here, more than the mean._")


def read(seed):
    log = os.path.join(HERE, f"cell_s{seed}.log")
    cpu = os.path.join(HERE, f"mhl_sac_c50_s{seed}_cpueval.json")
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
        return "done", 100.0, (f"trained, best MJX {float(dn.group(1)):.0f} · "
                               f"awaiting CPU eval")
    ms = ITER.findall(txt)
    if not ms:
        return "run", 0.0, "starting"
    it, tot, ret, cov, digit, nosp, eff, K, tbkt, tcr, best, mins = ms[-1]
    pct = 100.0 * int(it) / max(1, int(tot))
    # ETA from the MOST RECENT interval, not from elapsed/iters. The first 500 iterations
    # are warmup -- rollout only, no updates -- and finish in ~3 min, so an
    # elapsed-over-iterations estimate reads far too optimistic.
    if len(ms) >= 2:
        d_it = int(it) - int(ms[-2][0])
        d_min = float(mins) - float(ms[-2][11])
        rate = d_min / d_it if d_it > 0 else float(mins) / max(1, int(it))
    else:
        rate = None
    eta = rate * (int(tot) - int(it)) if rate else None
    eta_s = f"~{eta/60:.1f}h left" if eta is not None else "ETA after iter 2×eval"
    return "run", pct, (f"{int(it):,}/{int(tot):,} · ({float(ret):.0f}) · "
                        f"digit {float(digit):.2f} · eff {float(eff):.1f}/{K} · "
                        f"cov {float(cov):.0f}% · {eta_s}")


def stat(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
        if len(vals) > 1 else float("nan")
    return m, sd, sum(1 for v in vals if v >= TAKEOFF)


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
    lines += ["", "_already banked, exp_c50 seeds 0/1/2:_ "
                  + " · ".join(f"`s{s}` {v:.0f}" for s, v in sorted(PRIOR.items()))]
    if n_cpu:
        new = [json.load(open(os.path.join(
            HERE, f"mhl_sac_c50_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m, sd, tk = stat(new)
        lines += ["", f"*new seeds ({len(new)}): {m:.0f}"
                  + (f" ± {sd:.0f}*" if len(new) > 1 else "*")
                  + f", takeoff {tk}/{len(new)}"]
        pool = new + list(PRIOR.values())
        pm, psd, ptk = stat(pool)
        lines.append(f"*POOLED n={len(pool)}: {pm:.0f} ± {psd:.0f}*, takeoff "
                     f"{ptk}/{len(pool)}  ·  c36 target 4246 ± 298 (3/3)  ·  "
                     f"c49 baseline 2233 ± 1259 (1/3)")
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy. Takeoff threshold is the "
                  "chapter's usual *3000*. `digit` = mean hard bucket digit on the *0–15* "
                  "scale (one detector, 16 buckets), comparable to c50 seeds 0–2 "
                  "(7.85–9.21) and c49 (8.60–9.70). `eff` = 2^entropy of per-table cell "
                  "occupancy out of *16* cells (c50 seeds 0–2: 3.78–4.56). Delay "
                  "distributions for all six are read off the checkpoints at the end and "
                  "reported against c36's._"]
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c50b — six more seeds (3–8) of the floor-removed clamp, to n=9 pooled",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c50b.handle"), "w").write(h)
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
                final_text=(f"exp_c50b — six more seeds "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
