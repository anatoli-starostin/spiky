"""exp_c32b — self-updating Slack bar for the 3-seed Bucket-LIF SAC sweep (#75).

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network
call, so this runs under `sbox` and costs no approval.

TWO NUMBERS, NEVER CONFLATED. During training a cell shows the MJX proxy in parentheses --
20 episodes, horizon 1000, perturbation-free physics. That is a watching number; c21 read
425 at iter 1500 and finished at 5287. The number that decides the experiment is the
deterministic 100-episode CPU reference, shown bare once eval_bucket_cpu.py has written it.

`bkt` is the cold-start diagnostic and is the one to watch early. Non-firing neurons fold
into the LAST bucket, so this starts near 15.0 with a small spread: at init the torch
reference reaches only 7 of 16 buckets. A run still sitting at 15.0±0.5 by iteration 2,000
has a membrane that never crosses threshold and will not recover. Healthy looks like the
mean falling and the spread RISING (smoke: 14.9±0.8 -> 11.7±3.6).

Exits once SWEEP_DONE_C32B exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and posted a frozen number for nine hours.

Usage:
  python slack_bar_c32.py --task <BODY_TASK id> [--handle <existing>] [--interval 150]
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
                  r"\s+\|\s+bkt-cov\s+([\d.]+)%\s+\|\s+bkt\s+([\d.]+)±([\d.]+)\s+\|\s+"
                  r"Tbkt\s+([\d.]+)\s+\|\s+Tcr\s+([\d.]+)\s+\|\s+best\s+([-\d.]+)\s+\|"
                  r"\s+([\d.]+)m", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
FAIL = re.compile(r"Traceback|CUDA out of memory|Killed|MemoryError|XlaRuntimeError")

SEEDS = (0, 1, 2)
TOTAL = len(SEEDS)
SENTINEL = "SWEEP_DONE_C32B"
REF = ("_*CORRECTED* BucketLIFDetectorsMHL (branch @ 2a2795e0). Three fixes vs the "
       "version exp_c32 ran: synapses are now bounded EXCITATORY `w = 2·sigmoid(w_raw)` "
       "instead of free-signed, a hot init `w_raw ~ N(-2.2, 0.5)`, and the tau floor "
       "raised 1e-3 → 1.0. The first is the one that matters — with signed weights the "
       "membrane rarely reached theta_mem, so neurons never spiked and folded into the "
       "LAST bucket. Measured at init: *no-spike mass 0.97 → 0.030*, mean bucket "
       "14.97 → 7.58. Params unchanged: *7,840* = 1,696 front-end + 6,144 table. "
       "Parity vs the new torch reference: 40/40. Anchors: *exp_c32 BROKEN 1174 ± 517, "
       "0/300 full*; exp_c18 hyperplane 4308 ± 500 (6 seeds); c31 PureLIF 2951 ± 2109; "
       "c30b 4087 ± 991; c30 3931 ± 586._")


def read(seed):
    log = os.path.join(HERE, f"cell_s{seed}.log")
    cpu = os.path.join(HERE, f"bucket_sac_c32b_s{seed}_cpueval.json")
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
    it, tot, ret, cov, bm, bsd, tbkt, tcr, best, mins = ms[-1]
    pct = 100.0 * int(it) / max(1, int(tot))
    eta = (float(mins) / max(1, int(it))) * (int(tot) - int(it))
    return "run", pct, (f"{int(it):,}/{int(tot):,} · ({float(ret):.0f}) · "
                        f"bkt {float(bm):.1f}±{float(bsd):.1f} · "
                        f"cov {float(cov):.0f}% · Tbkt {float(tbkt):.2f} · "
                        f"~{eta/60:.1f}h left")


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
            HERE, f"bucket_sac_c32b_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        lines += ["", f"*mean over {len(vals)} seeds: {m:.0f}" +
                  (f" ± {sd:.0f}*" if len(vals) > 1 else "*") +
                  "  ·  c18 hyperplane 4308 ± 500 at 3.6× the params"]
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy. `bkt` = mean±sd bucket index; "
                  "starts ~15 (non-firing folds into the last bucket) and should FALL "
                  "while the spread RISES._"]
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c32b — CORRECTED Bucket-LIF, 16 buckets × 32 tables, 3 seeds (#75)",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c32b.handle"), "w").write(h)
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
                final_text=(f"exp_c32 Bucket-LIF sweep "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
