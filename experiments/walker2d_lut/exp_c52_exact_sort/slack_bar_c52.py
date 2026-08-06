"""exp_c52 — self-updating Slack bar for the exact-sort ablation.

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network
call, so this runs under `sbox` and costs no approval.

WHAT THIS RUN IS. exp_c50 with one module constant flipped: SORT_FORM "rank" -> "argsort",
putting back the `jnp.argsort(a, stable=True)` + `take_along_axis` spelling that exp_c36
used. Same three seeds, everything else identical.

WHY THE BAR SHOWS AN EXPECTED VALUE PER SEED. `sort_equivalence.py` established before this
sweep started that the two forms are bit-identical -- including on a constructed case where
100% of adjacent arrival pairs are exactly tied -- so the training trajectory should match
c50's exactly and each seed should land on its c50 value to the digit. That makes this an
unusually strong bar: any deviation at all is informative, so the deltas are displayed
rather than just the numbers.

IT IS SLOW, BY DESIGN OF THE THING BEING TESTED. c36's argsort spelling cost 240.5 min per
seed against c50's 37.4: `jnp.argsort` alone is 19-22 ms and its `take_along_axis` VJP is a
scatter-add that `--xla_gpu_deterministic_ops=true` serialises. Expect ~4 hours.

Exits once SWEEP_DONE_C52 exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and posted a frozen number for nine hours.

Usage:
  python slack_bar_c52.py --task <BODY_TASK id> [--handle <existing>] [--interval 300]
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
EXPECT = {0: 4447.2, 1: 3719.6, 2: 1156.3}      # exp_c50, which this should reproduce
TAKEOFF = 3000.0
SENTINEL = "SWEEP_DONE_C52"
REF = ("_*exp_c52 — the exact-sort ablation.* exp_c50 with ONE module constant flipped: "
       "`SORT_FORM` 'rank' → *'argsort'*, restoring the `jnp.argsort(a, stable=True)` + "
       "`take_along_axis` spelling exp_c36 used (`jax_bucket_lif.py:207`). Everything else "
       "is c50 verbatim — 1 head × *128 tables × 1 detector × 16 buckets*, per-table betas, "
       "stock 0.1 table init, `delay_init_std=0`, temperatures trainable, delay clamp with "
       "the LOWER bound removed and the upper cap kept, seeds 0/1/2. Parity *105/105*.\n"
       "_*The answer is already known, and this measures it anyway.* Before the sweep "
       "started, `sort_equivalence.py` compared the two forms on identical weights: every "
       "intermediate (`a_srt`, `t_hard`, `t_soft`, the soft partition), both forwards, the "
       "bucket digits, the cell index and the gradient of *all 8 parameters* agree to "
       "exactly *0.000e+00* — on a fresh init, on perturbed random weights, on all three "
       "trained c50 checkpoints, and on a constructed case where *100% of adjacent arrival "
       "pairs are exactly tied* so the tie-break alone decides the permutation. 0 of 8,192 "
       "digits differ in every case._\n"
       "_*So the prediction is exact:* identical gradients ⇒ identical trajectory ⇒ each "
       "seed lands on its c50 value to the digit — *s0 4447.2, s1 3719.6, s2 1156.3*. The "
       "bar shows the delta against that, because at this strength of prediction any "
       "nonzero deviation is the interesting event._\n"
       "_*Cost.* This spelling is why `rank` exists: c36 took *240.5 min/seed* against "
       "c50's *37.4*. `jnp.argsort` alone is 19–22 ms, and its `take_along_axis` VJP is a "
       "scatter-add that `--xla_gpu_deterministic_ops=true` serialises. Expect *~4 hours*, "
       "not 40 minutes._\n"
       "_*What it settles.* If the seeds reproduce, rank-vs-sort is eliminated as a "
       "candidate for the residual c50-vs-c36 gap and the bisect moves to the membrane "
       "formulation and the bucket-digit path. If they do NOT, the bit-identity result "
       "above is wrong in a way worth understanding immediately._")


def read(seed):
    log = os.path.join(HERE, f"cell_s{seed}.log")
    cpu = os.path.join(HERE, f"mhl_sac_c52_s{seed}_cpueval.json")
    if os.path.exists(cpu):
        d = json.load(open(cpu))
        v = d["cpu_reference_mean"]
        dv = v - EXPECT[seed]
        tag = "*exact match*" if abs(dv) < 0.05 else f"*Δ {dv:+.1f} vs c50*"
        return "cpu", 100.0, (f"*{v:.1f}* ±{d['cpu_reference_std']:.0f} "
                              f"({d['full_length']}/100 full) · {tag}")
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
    # ETA from the MOST RECENT interval, not from elapsed/iters -- the first 500 iterations
    # are warmup (rollout only, no updates) and badly under-estimate the real rate.
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
        lines.append(f"{mark} `seed {s}` {text}   _(c50 predicts {EXPECT[s]:.1f})_")
    if n_cpu:
        vals = [json.load(open(os.path.join(
            HERE, f"mhl_sac_c52_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        lines += ["", f"*mean over {len(vals)} seeds: {m:.1f}"
                  + (f" ± {sd:.1f}*" if len(vals) > 1 else "*")
                  + "  ·  c50 (same seeds) 3107.7 ± 1728.7, takeoff 2/3  ·  "
                    "c36 4246 ± 298 (3/3)"]
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy. Shown to one decimal here "
                  "because the prediction is exact equality with c50, not approximate "
                  "agreement. `digit` = mean hard bucket digit on the *0–15* scale, `eff` "
                  "= 2^entropy of per-table cell occupancy out of *16* cells; both should "
                  "also track c50's seeds 0–2 (digit 7.85–9.21, eff 3.78–4.56) step for "
                  "step if the bit-identity result holds._"]
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c52 — exact-sort ablation (rank → argsort), 3 seeds, ~4h",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c52.handle"), "w").write(h)
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
                final_text=(f"exp_c52 exact-sort ablation "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
