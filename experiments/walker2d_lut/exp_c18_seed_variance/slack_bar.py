"""exp_c18 — live Slack bar for the 6-seed variance study (#75).

Two things this bar does that a naive one would not:

  1. It covers BOTH stages. Training is ~35 min a run and the eval that produces the
     actual number is ~10 s, so each seed is weighted 90/10 and the bar refuses to
     finalise until all six evals have written their JSON.
  2. It checks for FAILURE BEFORE reporting progress. A run that died at 9,000 iters
     still has a healthy-looking last progress line; reading the traceback first is what
     stops a dead run from being displayed at 90%.
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+([\d.]+)%\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
SEEDS = (0, 1, 2, 3, 4, 5)
TRAIN_W = 0.9
REF = ("hyperplane×hard · anchor_pairs init · nap6/tph32 · 10k iters · determinism ON "
       "(exp_c17 fix) so every seed is bit-reproducible · 3 concurrent, 2 waves")


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def read(seed):
    ev = os.path.join(C09, f"lut_sac_c18_seed{seed}_cpueval.json")
    st = dict(seed=seed, eval_done=os.path.exists(ev), cpu=None)
    if st["eval_done"]:
        try:
            st["cpu"] = json.load(open(ev))["cpu_reference_mean"]
        except Exception:
            st["eval_done"] = False
    try:
        txt = open(os.path.join(HERE, f"cell_seed{seed}.log"), errors="replace").read()
    except OSError:
        st.update(state="queued", pct=0.0)
        return st
    d = DONE.search(txt)
    if re.search(r"Traceback|CUDA out of memory|Killed|RESOURCE_EXHAUSTED", txt) and not d:
        st.update(state="failed", pct=0.0)
        return st
    rows = ITER.findall(txt)
    if d:
        st.update(state="trained", pct=100.0, best=float(d.group(1)),
                  cov=float(rows[-1][3]) if rows else 0.0)
    elif rows:
        it, tot, _ret, cov, best = rows[-1]
        st.update(state="training", pct=100.0 * int(it) / int(tot), it=int(it),
                  tot=int(tot), best=float(best), cov=float(cov))
    else:
        st.update(state="starting", pct=0.0)
    return st


def render():
    sts = [read(s) for s in SEEDS]
    frac = sum(TRAIN_W * s["pct"] / 100.0
               + (1 - TRAIN_W) * (1.0 if s["eval_done"] else 0.0)
               for s in sts) / len(sts)
    n_fail = sum(1 for s in sts if s["state"] == "failed")
    n_eval = sum(1 for s in sts if s["eval_done"])
    n_train = sum(1 for s in sts if s["state"] in ("trained",) or s["eval_done"])
    done_all = n_eval == len(sts)
    running = sum(1 for s in sts if s["state"] in ("training", "starting"))

    head = ("all 6 seeds complete" if done_all
            else f"{n_train}/6 trained · {n_eval}/6 evaluated · {running} on GPU now")
    lines = [f"*{head}*" + (f" · *{n_fail} FAILED*" if n_fail else "")]
    for s in sts:
        lab = f"seed {s['seed']}"
        if s["eval_done"]:
            lines.append(f"`{lab}` {bar(100)} ✅ CPU-ref *{s['cpu']:.1f}*")
        elif s["state"] == "trained":
            lines.append(f"`{lab}` {bar(100)} trained (best MJX {s['best']:.0f}) · "
                         f"eval pending…")
        elif s["state"] == "training":
            lines.append(f"`{lab}` {bar(s['pct'])} {s['pct']:3.0f}% · "
                         f"{s['it']:,}/{s['tot']:,} · best MJX {s['best']:.0f} · "
                         f"row-cov {s['cov']:.0f}%")
        elif s["state"] == "failed":
            lines.append(f"`{lab}` {bar(0)} ❌ failed")
        else:
            lines.append(f"`{lab}` {bar(0)}  queued (waits for a free slot)")

    if done_all:
        import statistics
        v = [s["cpu"] for s in sts]
        lines.append(f"\n*6-seed mean {statistics.mean(v):.1f} ± "
                     f"{statistics.stdev(v):.1f}* · range {min(v):.1f}–{max(v):.1f} "
                     f"(spread {max(v) - min(v):.0f})")
    return 100.0 * frac, done_all, n_fail, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=180)
    a = ap.parse_args()
    h = progress.progress_start("seed-variance study — 6 deterministic seeds (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c18.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        pct, done_all, n_fail, body = render()
        if done_all or n_fail == len(SEEDS):
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=(n_fail == 0),
                final_text=(body + "\n_full analysis follows in-thread_\n"
                            f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=f"_{REF}_\n{body}")
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}%", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
