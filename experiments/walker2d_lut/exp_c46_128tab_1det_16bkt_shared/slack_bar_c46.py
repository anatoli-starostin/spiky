"""exp_c46 — self-updating Slack bar for the 3-seed MHL-LIF SAC sweep.

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network
call, so this runs under `sbox` and costs no approval.

TWO NUMBERS, NEVER CONFLATED. During training a cell shows the MJX proxy in parentheses --
20 episodes, horizon 1000, perturbation-free physics. That is a watching number; c21 read
425 at iteration 1,500 and finished at 5,287. The number that decides the experiment is the
deterministic 100-episode CPU reference, shown bare once eval_mhl_cpu.py has written it.

WHAT TO WATCH EARLY. 128 tables, ONE detector each, 16 ordered buckets, and -- the handicap
under test -- a SINGLE bucket ladder shared by all of them. A table's cell index is just its
detector's bucket digit, read off boundaries that every other table reads too:

  digit mean hard bucket digit over (sample, table), ranging 0..15. A detector that never
        spikes folds into the LAST bucket, so this starts near 15 and should FALL. Pinned
        anywhere at all means the detectors have collapsed to a constant and each table is
        one row. The SCALE differs in every experiment of this line -- 0..1 in c38, 0..3 in
        c39, 0..63 in c43, 0..31 in c44/c45, 0..15 here -- never comparable across them.
        exp_c45 is the useful contrast: sharing pushed `digit` UP to 28.2-28.6 of 31,
        because detectors could no longer place boundaries where their own spike-time
        distribution sat, so more fell past the top boundary. Watch for that signature
        here, scaled to 0..15.
  eff   2**entropy of the per-table cell occupancy, out of 16 cells here (this config has
        16 buckets on one detector). Every single-detector bucket configuration c32b-c37
        converged to 1.7-2.5 and c46 is one of those -- though exp_c43 was the fourth
        confirmation that this diagnostic does not predict takeoff, so it is context, not
        a leading indicator.

Exits once SWEEP_DONE_C46 exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and posted a frozen number for nine hours.

Usage:
  python slack_bar_c46.py --task <BODY_TASK id> [--handle <existing>] [--interval 150]
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
SENTINEL = "SWEEP_DONE_C46"
REF = ("_*exp_c46 — 128 detectors WITH the shared-ladder handicap.* Config: 1 head × "
       "*128 tables × 1 LIF detector × 16 buckets*, freeze_temperature=True, "
       "delay_init_std=4, table_init_std = 0.1/√128 = *0.0088388* (recomputed for "
       "tph=128, not inherited from c45), and `share_betas=True` — ONE global ladder, "
       "`beta_base` (1,1,1) + `beta_raw` (1,1,15), broadcast to all 128 tables.\n"
       "_*29,328 params* (29,072 trainable) = 4,752 front-end + 24,576 table = *104.6%* "
       "of the 28,032 hyperplane baseline. Sharing collapses beta to *16* parameters — "
       "0.3% of the front-end, against c36's 2,048 for the same 128 tables × 16 buckets. "
       "Parity *90/90*, including byte-identical ladders across all 128 tables (max "
       "spread 0.0) and that the shared ladder reaches the forward._\n"
       "_*Two questions this settles.* (a) *vs c45* (*1198 ± 940*, 0/3): same shared "
       "handicap, DOUBLE the detectors (64 → 128) — does detector count recover the "
       "shared-ladder penalty? (b) *vs c36* (*4246 ± 298*, the only config in this line "
       "ever to reach the baseline): the SAME 128 detectors and the same 128 tables × 16 "
       "buckets, but c36 gave every table its own ladder. So c46 measures the "
       "shared-ladder penalty at the one configuration that actually works._\n"
       "_Context: exp_c45 found sharing cost ~1,163 return at 64 detectors, hurt all 3 "
       "seeds paired, and dropped ρ(detector count) from +0.81 to +0.75 — detector count "
       "orders this line but is not sufficient._\n"
       "⚠️ _n=3 cannot resolve takeoff-rate differences (the c42b lesson): read the MEANS "
       "against c45 and c36, treat the takeoff count as indicative. Baseline: exp_c18 "
       "*4308 ± 500* (6 seeds)._")


def read(seed):
    log = os.path.join(HERE, f"cell_s{seed}.log")
    cpu = os.path.join(HERE, f"mhl_sac_c46_s{seed}_cpueval.json")
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
            HERE, f"mhl_sac_c46_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        lines += ["", f"*mean over {len(vals)} seeds: {m:.0f}" +
                  (f" ± {sd:.0f}*" if len(vals) > 1 else "*") +
                  "  ·  c18 hyperplane 4308 ± 500 at 0.96× these params"]
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy. `digit` = mean hard bucket "
                  "digit, which here ranges *0–15* (one detector, 16 buckets) — a "
                  "non-firing detector folds into the LAST bucket so it starts near 15 "
                  "and should FALL. Different scale again from c44/c45's 0–31, c43's "
                  "0–63, c39's 0–3 and c38's 0–1; none are comparable. `eff` = 2^entropy "
                  "of per-table cell occupancy, out of *16* cells here — every "
                  "single-detector bucket config c32b–c37 sat at 1.7–2.5 and c46 is one, "
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
        "exp_c46 — 128 tables × 1 detector × 16 buckets, shared ladder, 3 seeds",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c46.handle"), "w").write(h)
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
                final_text=(f"exp_c46 MHL-LIF sweep "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
