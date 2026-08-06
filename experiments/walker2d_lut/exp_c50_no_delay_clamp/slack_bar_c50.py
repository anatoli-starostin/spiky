"""exp_c50 — self-updating Slack bar for the 3-seed delay-clamp-removal sweep.

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network
call, so this runs under `sbox` and costs no approval.

TWO NUMBERS, NEVER CONFLATED. During training a cell shows the MJX proxy in parentheses --
20 episodes, horizon 1000, perturbation-free physics. That is a watching number; c21 read
425 at iteration 1,500 and finished at 5,287. The number that decides the experiment is the
deterministic 100-episode CPU reference, shown bare once eval_mhl_cpu.py has written it.

WHAT TO WATCH EARLY. This is c49's exact configuration -- 128 tables, ONE detector each, 16
ordered buckets, per-table ladders, stock 0.1 table init, temperatures trainable -- with one
module constant changed. `digit` therefore runs on the SAME 0-15 scale as c46/c47/c48/c49
and is directly comparable to them (c49 ran 8.60-9.70, c48 11.61-12.45), and `eff` is out of
16 cells (c49 3.60-4.19, c48 1.80-2.97). Neither is a leading indicator of takeoff -- c43
was the fourth confirmation of that -- so they are context, not a verdict. The diagnostic
this experiment exists for is the end-of-training DELAY distribution, read off the
checkpoints once the sweep finishes.

Exits once SWEEP_DONE_C50 exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and posted a frozen number for nine hours.

Usage:
  python slack_bar_c50.py --task <BODY_TASK id> [--handle <existing>] [--interval 150]
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
                  r"\s+\|\s+cell-cov\s+([\d.]+)%\s+\|\s+digit\s+([\d.]+)\s+\|\s+"
                  r"nosp\s+([\d.]+)\s+\|\s+eff\s+([\d.]+)/(\d+)\s+\|\s+"
                  r"Tbkt\s+([\d.]+)\s+\|\s+Tcr\s+([\d.]+)\s+\|\s+best\s+([-\d.]+)\s+\|"
                  r"\s+([\d.]+)m", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
FAIL = re.compile(r"Traceback|CUDA out of memory|Killed|MemoryError|XlaRuntimeError")

SEEDS = (0, 1, 2)
TOTAL = len(SEEDS)
SENTINEL = "SWEEP_DONE_C50"
REF = ("_*exp_c50 — the delay clamp's non-negativity floor REMOVED.* exp_c49 verbatim — "
       "current unified `LIFMultiHeadLUT`, 1 head × *128 tables × 1 LIF detector × 16 "
       "buckets*, per-table betas, stock table init 0.1, `delay_init_std=0`, "
       "SORT_FORM='rank', temperatures trainable, seeds 0/1/2, 31,360 params — with ONE "
       "change: `clamp(delay, 0, t_window)` → `clamp(delay, −inf, t_window)`.\n"
       "_*Only the LOWER bound is dropped.* The upper `t_window` cap is kept: it is what "
       "holds arrivals inside [·, 2·t_window] so `exp(a/tau)` stays float32-safe in the "
       "reference's cumsum membrane, and removing it would confound the test with an "
       "overflow risk. Negative delays are exactly what upstream's floor forbade and what "
       "the old `BucketLIFDetectorsMHL` allowed._\n"
       "_*Parity 105/105*, including the decisive check: *1148/1148* negative delays now "
       "carry nonzero gradient, where upstream's floor gives exactly 0 for all of them — "
       "while the 1 delay above the retained cap is still correctly dead._\n"
       "_*What this is testing.* c49 ends with *94.6–94.9%* of its 2,176 delays at or "
       "below the floor — clamped in the forward AND carrying exactly zero gradient, so "
       "permanently dead; delay capacity collapses to ~100 live parameters. c36, on the "
       "old UNCLAMPED module, ends ~40% negative and fully functional, spanning −10.08 … "
       "+12.67. A negative delay is not unphysical: the timeline origin is arbitrary, so "
       "a global shift renormalises the minimum to zero while preserving every relative "
       "arrival._\n"
       "_• *c50 ≈ 4246* → the clamp is the WHOLE gap; the refactor is otherwise clean, "
       "and this is an upstream finding for nucstar._\n"
       "_• *c50 ≈ 2233* → the clamp is not the cause either; the bisect escalates to the "
       "membrane formulation and the bucket-digit path._\n"
       "⚠️ _n=3 cannot resolve takeoff-rate differences (the c42b lesson) — read the "
       "MEAN. c49's baseline is *2233 ± 1259* (1/3); c36's target is *4246 ± 298* (3/3)._")


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
    # are warmup -- rollout only, no updates -- and finish in 2.8 min, so an
    # elapsed-over-iterations estimate reads ~0.9 h for a run that actually takes ~4 h.
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
            HERE, f"mhl_sac_c50_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        lines += ["", f"*mean over {len(vals)} seeds: {m:.0f}" +
                  (f" ± {sd:.0f}*" if len(vals) > 1 else "*") +
                  "  ·  c49 baseline 2233  ·  c36 target 4246  ·  "
                  "c18 hyperplane 4308 ± 500"]
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy. `digit` = mean hard bucket "
                  "digit on the *0–15* scale (one detector, 16 buckets), directly "
                  "comparable to c49 (8.60–9.70) and c48 (11.61–12.45), not to c44/c45's "
                  "0–31, c43's 0–63, c39's 0–3 or c38's 0–1. `eff` = 2^entropy of "
                  "per-table cell occupancy out of *16* cells (c49 3.60–4.19, c48 "
                  "1.80–2.97). The end-of-training DELAY distribution — range, % "
                  "negative, % on the retained upper cap — is the diagnostic this "
                  "experiment exists for, and is read off the checkpoints at the end._"]
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c50 — delay clamp's non-negativity floor removed, 3 seeds",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c50.handle"), "w").write(h)
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
                final_text=(f"exp_c50 delay-clamp-removal sweep "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
