"""exp_c17 — self-updating Slack bar for the replicate test (#75).

Unlike the sweep bars, this one has to cover TWO STAGES per run: training and then the
CPU-reference eval. A bar that stopped at "training done" would go quiet during the part
that produces the actual number. So each replicate is weighted 90% train / 10% eval, and
the bar only finalises when both evals have written their JSON.
"""
import argparse, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+([\d.]+)%\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
REPS = ("a", "b")
TRAIN_W = 0.9          # training is the long pole; eval is ~10 s against ~27 min
REF = ("XLA_FLAGS=--xla_gpu_deterministic_ops=true · without it exp_c16 gave |A−B| = 999.1 at this same seed · the real test is checkpoint identity")


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def read(rep):
    import json
    ev = os.path.join(C09, f"lut_sac_c17_det_{rep}_cpueval.json")
    st = dict(rep=rep, eval_done=os.path.exists(ev), cpu=None)
    if st["eval_done"]:
        try:
            st["cpu"] = json.load(open(ev))["cpu_reference_mean"]
        except Exception:
            st["eval_done"] = False
    p = os.path.join(HERE, f"cell_det_{rep}.log")
    try:
        txt = open(p, errors="replace").read()
    except OSError:
        st.update(state="queued", pct=0.0)
        return st
    d = DONE.search(txt)
    rows = ITER.findall(txt)
    # failure before progress, so a run that died at 9,000 iters is not shown as healthy
    if re.search(r"Traceback|CUDA out of memory|Killed", txt) and not d:
        st.update(state="failed", pct=0.0)
        return st
    if d:
        st.update(state="trained", pct=100.0, best=float(d.group(1)),
                  cov=float(rows[-1][3]) if rows else 0.0)
        return st
    if rows:
        it, tot, ret, cov, best = rows[-1]
        st.update(state="training", pct=100.0 * int(it) / int(tot), it=int(it),
                  tot=int(tot), best=float(best), cov=float(cov))
        return st
    st.update(state="starting", pct=0.0)
    return st


def render():
    sts = [read(r) for r in REPS]
    # per-replicate completion = 90% training + 10% eval
    frac = sum(TRAIN_W * s["pct"] / 100.0 + (1 - TRAIN_W) * (1.0 if s["eval_done"] else 0.0)
               for s in sts) / len(REPS)
    n_fail = sum(1 for s in sts if s["state"] == "failed")
    done_all = all(s["eval_done"] for s in sts)
    active = next((i for i, s in enumerate(sts)
                   if s["state"] in ("training", "starting")), None)

    head = ("both replicates complete" if done_all
            else (f"run {active + 1} of 2 training" if active is not None
                  else "evaluating"))
    lines = [f"*{head}*" + (f" · *{n_fail} FAILED*" if n_fail else "")]
    for i, s in enumerate(sts):
        lab = f"replicate {s['rep']} (seed 0)"
        if s["eval_done"]:
            lines.append(f"`{lab}` {bar(100)} ✅ CPU-ref *{s['cpu']:.1f}*")
        elif s["state"] == "trained":
            lines.append(f"`{lab}` {bar(100)} trained (best MJX {s['best']:.0f}) · "
                         f"eval pending…")
        elif s["state"] == "training":
            lines.append(f"`{lab}` {bar(s['pct'])} {s['pct']:3.0f}% · "
                         f"{s['it']:,}/{s['tot']:,} · best MJX {s['best']:.0f}")
        elif s["state"] == "failed":
            lines.append(f"`{lab}` {bar(0)} ❌ failed")
        else:
            lines.append(f"`{lab}` {bar(0)}  queued (sequential — waits for run 1)")
    if all(s["eval_done"] for s in sts):
        d = abs(sts[0]["cpu"] - sts[1]["cpu"])
        lines.append(f"\n*|A − B| = {d:.1f}* — return gap with determinism forced ON "
                     f"(exp_c16 without it: 999.1)")
    return 100.0 * frac, done_all, n_fail, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=180)
    a = ap.parse_args()
    h = progress.progress_start("determinism test — deterministic GPU ops (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c17.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        pct, done_all, n_fail, body = render()
        stats = f"_{REF}_\n{body}"
        if done_all or n_fail == len(REPS):
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=(n_fail == 0),
                final_text=(body + f"\n_full analysis follows in-thread_\n"
                                   f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}%", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
