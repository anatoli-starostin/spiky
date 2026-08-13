"""exp_c36b — self-updating Slack bar for the c36 anchor reproduction (seeds 3, 4, 5).

Cage-safe by construction: `progress.py` is a FILE RENDEZVOUS under ~/.cache, not a network
call, so this runs under `sbox` and costs no approval.

WHAT THIS RUN IS. Not a new configuration and not a port: exp_c36 continued on three fresh
seeds, using the SAME `bucket_sac.py` and `jax_bucket_lif.py` the original three ran, copied
unmodified. Old pre-refactor BucketLIFDetectorsMHL architecture, no delay clamp anywhere,
trainable temperatures, 1 head x 128 tables x 16 buckets.

WHY IT MATTERS MORE THAN A USUAL SWEEP. c36 is the anchor every experiment from c48 to c53
has been measured against. It is also n=3, in a lineage that is demonstrably bimodal. This
run tests the anchor itself rather than another challenger.

IT IS SLOW. c36 measured 240.5 min/seed with three co-resident, against 37.4 for the MHL
runs -- the old module's cumsum membrane and exact-argsort ordering are simply more
expensive. Expect ~4 hours, and note the bar's ETA only becomes meaningful after the second
eval (the first 500 iterations are warmup and badly under-estimate the rate).

Exits once SWEEP_DONE_C36B exists or --max-hours passes. Deliberately not `pgrep -f`:
exp_c25's refresher matched its own command line and posted a frozen number for nine hours.

Usage:
  python slack_bar_c36b.py --task <BODY_TASK id> [--handle <existing>] [--interval 300]
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
C36 = os.path.join(HERE, "..", "exp_c36_bucket_tables")
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+).*?"
                  r"best\s+([-\d.]+)\s+\|\s+([\d.]+)m", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
FAIL = re.compile(r"Traceback|CUDA out of memory|Killed|MemoryError|XlaRuntimeError")

SEEDS = (3, 4, 5)
TOTAL = len(SEEDS)
PRIOR = {0: 4527.5, 1: 3933.2, 2: 4277.6}       # the original exp_c36 three
TAKEOFF = 3000.0
SENTINEL = "SWEEP_DONE_C36B"
REF = ("_*exp_c36b — reproducing the ANCHOR itself.* Three more seeds (3, 4, 5) of "
       "exp_c36, using the same `bucket_sac.py` and `jax_bucket_lif.py` the original "
       "three ran, copied unmodified: old pre-refactor BucketLIFDetectorsMHL, *no delay "
       "clamp anywhere*, trainable temperatures, 1 head × *128 tables × 16 buckets*, "
       "31,360 params. Config is c36's own (taken from its run JSONs, which match the "
       "trainer defaults exactly). Parity *40/40* before any GPU.\n"
       "_*Why this and not another challenger.* c36 — *4246.1 ± 298.4, 3/3 takeoff* — is "
       "the anchor every experiment from c48 to c53 has been scored against, and the whole "
       "'residual gap' narrative rests on it. But it is n=3, and this lineage is bimodal: "
       "exp_c50 pooled to n=9 measures a takeoff rate of *4/9 = 0.444*, under which a 3/3 "
       "result has probability 0.444³ ≈ *8.8%*. Unlikely, not negligible. If c36 drew a "
       "lucky three, part of the gap the bisect is chasing does not exist._\n"
       "_*What the outcomes mean:*_\n"
       "_• *pooled n=6 stays near 4246, 6/6 or 5/6* → the anchor is solid, the gap is "
       "real, and the bisect continues to the membrane formulation and the bucket-digit "
       "path._\n"
       "_• *pooled n=6 falls toward c50's 2700* → c36's 3/3 was a lucky draw, the old and "
       "new modules are much closer than they looked, and the c48–c53 shortfall narrative "
       "needs revising from the ground up._\n"
       "⚠️ _*~4 hours*, not 40 minutes: c36 measured *240.5 min/seed* with three "
       "co-resident (the old cumsum membrane and exact argsort), against 37.4 for the MHL "
       "runs. ETA is only meaningful after the second eval._")


def read(seed):
    log = os.path.join(HERE, f"cell_s{seed}.log")
    cpu = os.path.join(HERE, f"bucket_sac_c36_s{seed}_cpueval.json")
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
    it, tot, ret, best, mins = ms[-1]
    pct = 100.0 * int(it) / max(1, int(tot))
    # ETA from the MOST RECENT interval, not elapsed/iters: the first 500 iterations are
    # warmup (rollout only, no updates) and badly under-estimate the real rate.
    if len(ms) >= 2:
        d_it = int(it) - int(ms[-2][0])
        d_min = float(mins) - float(ms[-2][4])
        rate = d_min / d_it if d_it > 0 else float(mins) / max(1, int(it))
    else:
        rate = None
    eta = rate * (int(tot) - int(it)) if rate else None
    eta_s = f"~{eta/60:.1f}h left" if eta is not None else "ETA after iter 2×eval"
    return "run", pct, (f"{int(it):,}/{int(tot):,} · ({float(ret):.0f}) · "
                        f"best {float(best):.0f} · {float(mins):.0f}m in · {eta_s}")


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
    lines += ["", "_the original exp_c36 three:_ "
                  + " · ".join(f"`s{s}` {v:.0f}" for s, v in sorted(PRIOR.items()))
                  + "  → *4246 ± 298, 3/3*"]
    if n_cpu:
        new = [json.load(open(os.path.join(
            HERE, f"bucket_sac_c36_s{s}_cpueval.json")))["cpu_reference_mean"]
            for s in SEEDS if cells[s][0] == "cpu"]
        m, sd, tk = stat(new)
        lines += ["", f"*new seeds ({len(new)}): {m:.0f}"
                  + (f" ± {sd:.0f}*" if len(new) > 1 else "*")
                  + f", takeoff {tk}/{len(new)}"]
        pool = new + list(PRIOR.values())
        pm, psd, ptk = stat(pool)
        lines.append(f"*POOLED c36 n={len(pool)}: {pm:.0f} ± {psd:.0f}*, takeoff "
                     f"{ptk}/{len(pool)}  ·  c50 pooled n=9 *2700 ± 1394* (4/9)")
    lines += ["", "_bare number = 100-ep deterministic CPU reference (the result); "
                  "(parenthesised) = live 20-ep MJX proxy. Takeoff threshold is the "
                  "chapter's usual *3000*. The decisive comparison is the POOLED c36 n=6 "
                  "against c50's pooled n=9 (2700 ± 1394) — if those two overlap, the "
                  "module refactor never cost what it appeared to._"]
    return pct, n_done, n_fail, n_cpu, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()
    h = a.handle or progress.progress_start(
        "exp_c36b — reproducing the c36 anchor, 3 more seeds (~4h)",
        task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c36b.handle"), "w").write(h)
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
                final_text=(f"exp_c36b anchor reproduction "
                            f"{'finished' if finished else 'TIMED OUT'} {stamp} — "
                            f"{n_done}/{TOTAL} trained, {n_cpu}/{TOTAL} CPU-evaluated"
                            + (f", {n_fail} failed" if n_fail else "")
                            + "\n" + body))
            print(f"bar closed at {stamp}", flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
