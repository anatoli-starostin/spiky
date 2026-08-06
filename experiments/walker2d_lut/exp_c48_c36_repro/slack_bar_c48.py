"""exp_c48 — self-updating Slack bar for the 3-seed MHL-LIF SAC sweep.

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network
call, so this runs under `sbox` and costs no approval.

TWO NUMBERS, NEVER CONFLATED. During training a cell shows the MJX proxy in parentheses --
20 episodes, horizon 1000, perturbation-free physics. That is a watching number; c21 read
425 at iteration 1,500 and finished at 5,287. The number that decides the experiment is the
deterministic 100-episode CPU reference, shown bare once eval_mhl_cpu.py has written it.

WHAT TO WATCH EARLY. 128 tables, ONE detector each, 16 ordered buckets, and -- restored
here -- a SEPARATE bucket ladder per table. A table's cell index is its detector's bucket
digit, read off that table's OWN boundaries:

  digit mean hard bucket digit over (sample, table), ranging 0..15. A detector that never
        spikes folds into the LAST bucket, so this starts near 15 and should FALL. Pinned
        anywhere at all means the detectors have collapsed to a constant and each table is
        one row. The SCALE differs in every experiment of this line -- 0..1 in c38, 0..3 in
        c39, 0..63 in c43, 0..31 in c44/c45, 0..15 in c46 and here -- never comparable
        across them, EXCEPT that c46 is the direct control: it ran this exact shape with a
        shared ladder and `digit` pinned high at 12.5-13.6 of 15, because detectors could
        not place boundaries where their own spike-time distribution sat. With per-table
        ladders restored, this should sit LOWER.
  eff   2**entropy of the per-table cell occupancy, out of 16 cells here. Every
        single-detector bucket configuration c32b-c37 converged to 1.7-2.5 and c47 is one
        of those -- though exp_c43 was the fourth confirmation that this diagnostic does
        not predict takeoff, so it is context, not a leading indicator.

Exits once SWEEP_DONE_C48 exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and posted a frozen number for nine hours.

Usage:
  python slack_bar_c48.py --task <BODY_TASK id> [--handle <existing>] [--interval 150]
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
SENTINEL = "SWEEP_DONE_C48"
REF = ("_*exp_c48 — c36 reproduction on the CURRENT module.* 1 head × *128 tables × 1 LIF "
       "detector × 16 buckets*, per-table betas, freeze_temperature=True, "
       "SORT_FORM='rank' — the current unified `LIFMultiHeadLUT`, NOT a rollback — with "
       "exactly two settings reverted to c36's: *stock table init 0.1* (deliberately NOT "
       "the fan-in 0.0088388) and *zero delays* (delay_init_std=0).\n"
       "_*31,360 params* (31,104 trainable) = 6,784 front-end + 24,576 table — identical "
       "to both c47 and the original c36. Parity *97/97*, and the gate EARNED ITS KEEP "
       "here: it caught a real port bug that only appears at delay_init_std=0. `jnp.clip` "
       "splits the gradient 0.5/0.5 at an exact tie while `torch.clamp` passes 1.0, so "
       "our delay gradient was exactly *2.000000×* the reference on all 2,176 entries. "
       "Fixed (rel 7.6e-02 → 1.6e-07). c38–c47 all used delay_init_std=4, whose delays "
       "are strictly positive, so the clamp was never active and this could not appear._\n"
       "_*Two comparisons.* (a) *vs c47* (*2784 ± 1744*, 2/3): same module, same shape, "
       "differing in EXACTLY stock-init + zero-delays — a clean 2-way isolation of the "
       "(fan-in + delays) effect. (b) *vs the original c36* (*4246 ± 298*: 4527.5 / "
       "3933.2 / 4277.6 on seeds 0/1/2): same 128×1×16 stock-init zero-delay per-table "
       "recipe._\n"
       "⚠️ _On (b) — c48 differs from c36 in TWO ways, not one. Besides the module, c36's "
       "temperatures were TRAINABLE and annealed hard (T_bkt 1.000 → *0.018*, T_cross "
       "→ 0.436 over its run); c48 has freeze_temperature=True, pinning both at 1.000. So "
       "a gap here is NOT attributable to the refactor alone. n=3 also cannot resolve "
       "takeoff-rate differences (the c42b lesson)._")


def read(seed):
    log = os.path.join(HERE, f"cell_s{seed}.log")
    cpu = os.path.join(HERE, f"mhl_sac_c48_s{seed}_cpueval.json")
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
    # Once two evals exist the slope between them is the real post-warmup rate.
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
            HERE, f"mhl_sac_c48_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        lines += ["", f"*mean over {len(vals)} seeds: {m:.0f}" +
                  (f" ± {sd:.0f}*" if len(vals) > 1 else "*") +
                  "  ·  c18 hyperplane 4308 ± 500 at 0.89× these params"]
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy. `digit` = mean hard bucket "
                  "digit, which here ranges *0–15* (one detector, 16 buckets) — a "
                  "non-firing detector folds into the LAST bucket so it starts near 15 "
                  "and should FALL. Directly comparable to *c47* (same 0–15 scale, same "
                  "shape and ladders, but fan-in init + delays) where it ran 10.99–12.17, "
                  "and to c46 (shared ladder) at 12.5–13.6. Not comparable to c44/c45's "
                  "0–31, c43's 0–63, c39's 0–3 or c38's 0–1. `eff` = 2^entropy of "
                  "per-table cell occupancy, out of *16* cells here — every "
                  "single-detector bucket config c32b–c37 sat at 1.7–2.5 and c48 is one, "
                  "so that band is the one to watch it against._"]
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c48 — c36 reproduction on the CURRENT module: stock init, zero delays, 3 seeds",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c48.handle"), "w").write(h)
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
                final_text=(f"exp_c48 MHL-LIF sweep "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
