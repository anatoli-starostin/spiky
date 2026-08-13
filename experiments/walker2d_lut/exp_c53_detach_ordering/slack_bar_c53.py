"""exp_c53 — self-updating Slack bar for the detached-hard-crossing sweep.

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network
call, so this runs under `sbox` and costs no approval.

WHAT THIS RUN IS. exp_c50 with `t_soft := t_hard`: the soft bucket partition is fed the
actual first-crossing arrival instead of the T_cross-weighted expectation over all N
arrivals. Same three seeds, everything else identical.

WHAT TO WATCH. `digit` and `eff` are on the same 0-15 / out-of-16 scales as c48-c52 and are
directly comparable to c50 seeds 0-2 (digit 7.85-9.21, eff 3.78-4.56). The new thing worth
watching is that `Tcr` should now sit PINNED at its initial 1.000 for the whole run: nothing
reads log_T_cross in this variant, so a moving Tcr would mean the detach did not take.

Exits once SWEEP_DONE_C53 exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and posted a frozen number for nine hours.

Usage:
  python slack_bar_c53.py --task <BODY_TASK id> [--handle <existing>] [--interval 150]
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
C50 = {0: 4447.2, 1: 3719.6, 2: 1156.3}         # same seeds, soft crossing
TAKEOFF = 3000.0
SENTINEL = "SWEEP_DONE_C53"
REF = ("_*exp_c53 — the DETACHED-HARD crossing.* exp_c50 with one change: `t_soft := "
       "t_hard`. The soft bucket partition is fed the ACTUAL first-crossing arrival "
       "instead of the T_cross-weighted expectation over all N arrivals. Everything else "
       "is c50 verbatim — 1 head × *128 tables × 1 detector × 16 buckets*, per-table "
       "betas, stock 0.1 table init, `delay_init_std=0`, delay clamp with the lower bound "
       "removed and the upper cap kept, SORT_FORM='rank', seeds 0/1/2. Parity *122/122*.\n"
       "_*First, a correction to the premise.* 'stop_gradient on the ordering' is a NO-OP "
       "in this module. Wrapping the permutation in an explicit `stop_gradient` changes no "
       "gradient in either variant — *0.000e+00* on all 8 parameters. The reorder decision "
       "was never differentiable: `rank` builds its permutation from integer comparisons, "
       "`argsort` from integer indices. What this variant actually removes is the SOFT "
       "CROSSING — the T_cross sigmoid-survival average._\n"
       "_*The case FOR it.* The soft partition IS the address-gradient path, and today its "
       "mode disagrees with the hard cell it stands in for *59% of the time*: argmax(g) == "
       "b_hard just *40.6%* under the soft crossing, against *99.2%* under detach_hard. "
       "`t_soft` sat a mean of *2.60* (max 23.2) from the real crossing. The gradient has "
       "been pointing at the wrong cell more often than the right one._\n"
       "_*The case AGAINST it.* It kills *2,432 parameters*. `w_raw` (2,176) and `tau_raw` "
       "(128) reach the output ONLY through the membrane potential V, and V now only picks "
       "a detached index — so synaptic weights and time constants stop learning entirely; "
       "`log_T_cross` (128) goes unused. That is *36% of the front-end*, dead by "
       "construction. Parity asserts all three dead on BOTH sides rather than leaving it "
       "to an autopsy._\n"
       "_*The question:* does a faithful address gradient over a crippled front-end beat "
       "an unfaithful one over a whole front-end? Reference points — c50 same seeds *3108 "
       "± 1729* (2/3), but pooled to n=9 c50 is *2700 ± 1394* (4/9); c36 *4246 ± 298* "
       "(3/3); c49 *2233 ± 1259* (1/3)._\n"
       "_*Speed, the other half.* Idle-5090 microbenchmark: the isolated `first_spike` "
       "value+grad is *1.33×* faster (the cumprod-survival VJP is the bulk), but "
       "whole-module value+grad is *1.01×* — 0.601 → 0.594 ms, ~0.2 ms of a ~220 ms "
       "iteration. End-to-end s/iter from this run against c50's ~0.22 settles it._\n"
       "⚠️ _`Tcr` should stay PINNED at 1.000 all run — nothing reads log_T_cross here, so "
       "a moving Tcr would mean the detach did not take._")


def read(seed):
    log = os.path.join(HERE, f"cell_s{seed}.log")
    cpu = os.path.join(HERE, f"mhl_sac_c53_s{seed}_cpueval.json")
    if os.path.exists(cpu):
        d = json.load(open(cpu))
        v = d["cpu_reference_mean"]
        return "cpu", 100.0, (f"*{v:.0f}* ±{d['cpu_reference_std']:.0f} "
                              f"({d['full_length']}/100 full) · "
                              f"_{v - C50[seed]:+.0f} vs c50_")
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
    # ETA from the MOST RECENT interval, not elapsed/iters: the first 500 iterations are
    # warmup (rollout only, no updates) and badly under-estimate the real rate.
    if len(ms) >= 2:
        d_it = int(it) - int(ms[-2][0])
        d_min = float(mins) - float(ms[-2][11])
        rate = d_min / d_it if d_it > 0 else float(mins) / max(1, int(it))
    else:
        rate = None
    eta = rate * (int(tot) - int(it)) if rate else None
    eta_s = f"~{eta/60:.1f}h left" if eta is not None else "ETA after iter 2×eval"
    s_it = f" · {60*rate/1:.3f}s/it" if rate else ""
    return "run", pct, (f"{int(it):,}/{int(tot):,} · ({float(ret):.0f}) · "
                        f"digit {float(digit):.2f} · eff {float(eff):.1f}/{K} · "
                        f"Tcr {float(tcr):.3f}{s_it} · {eta_s}")


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
            HERE, f"mhl_sac_c53_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        lines += ["", f"*mean over {len(vals)} seeds: {m:.0f}"
                  + (f" ± {sd:.0f}*" if len(vals) > 1 else "*")
                  + "  ·  c50 same seeds 3108 (n=9 pooled 2700)  ·  c36 4246  ·  "
                    "c49 2233"]
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy. `digit` = mean hard bucket "
                  "digit on the *0–15* scale, `eff` = 2^entropy of per-table cell "
                  "occupancy out of *16* — both directly comparable to c50 seeds 0–2 "
                  "(digit 7.85–9.21, eff 3.78–4.56). `Tcr` is shown because it must stay "
                  "pinned at 1.000 in this variant; `s/it` is the end-to-end speed "
                  "comparison against c50's ~0.22 s/iter._"]
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c53 — detached-hard crossing (t_soft := t_hard), 3 seeds",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c53.handle"), "w").write(h)
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
                final_text=(f"exp_c53 detached-hard-crossing sweep "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
