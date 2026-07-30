"""exp_c19 — live Slack bar for the MLP-actor control (#75).

A SECOND bar rather than an extension of exp_c18's. The LUT bar finalises when the LUT
study ends, and it should: mixing a queued control into a finished bar would either keep
that bar open for hours after its result is in, or let the control finish invisibly. This
bar therefore has an explicit WAITING state — the whole point of the run is that it does
not start until the GPU is free.
"""
import argparse, json, os, re, statistics, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C18 = os.path.join(HERE, "..", "exp_c18_seed_variance")
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+\S+\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
SEEDS = (0, 1, 2, 3, 4, 5)
TRAIN_W = 0.9
REF = ("MLP-actor SAC control · 2×256 · same env, same 6 seeds, same 10k iters, same "
       "determinism flags · queued behind the LUT study so it takes no GPU from it · "
       "measures SPREAD, not a ranking")


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def started():
    """The control is 'started' once its runner has stopped waiting on exp_c18."""
    try:
        return "starting the MLP control" in open(
            os.path.join(HERE, "run_seeds.log"), errors="replace").read()
    except OSError:
        return False


def read(seed):
    ev = os.path.join(HERE, f"mlp_sac_c19_seed{seed}_cpueval.json")
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
        st.update(state="trained", pct=100.0, best=float(d.group(1)))
    elif rows:
        it, tot, _ret, best = rows[-1]
        st.update(state="training", pct=100.0 * int(it) / int(tot), it=int(it),
                  tot=int(tot), best=float(best))
    else:
        st.update(state="starting", pct=0.0)
    return st


def render():
    if not started():
        return 0.0, False, 0, ("*waiting for the LUT study to finish* — the control is "
                               "queued deliberately so it takes no GPU from exp_c18.\n"
                               "`6 MLP seeds` " + bar(0) + "  not started")
    sts = [read(s) for s in SEEDS]
    frac = sum(TRAIN_W * s["pct"] / 100.0
               + (1 - TRAIN_W) * (1.0 if s["eval_done"] else 0.0)
               for s in sts) / len(sts)
    n_fail = sum(1 for s in sts if s["state"] == "failed")
    n_eval = sum(1 for s in sts if s["eval_done"])
    n_train = sum(1 for s in sts if s["state"] == "trained" or s["eval_done"])
    done_all = n_eval == len(sts)
    running = sum(1 for s in sts if s["state"] in ("training", "starting"))

    head = ("all 6 MLP seeds complete" if done_all
            else f"{n_train}/6 trained · {n_eval}/6 evaluated · {running} on GPU now")
    lines = [f"*{head}*" + (f" · *{n_fail} FAILED*" if n_fail else "")]
    for s in sts:
        lab = f"MLP seed {s['seed']}"
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
            lines.append(f"`{lab}` {bar(0)}  queued")

    if done_all:
        v = [s["cpu"] for s in sts]
        m, sd = statistics.mean(v), statistics.stdev(v)
        lines.append(f"\n*MLP 6-seed mean {m:.1f} ± {sd:.1f}* · range {min(v):.1f}–"
                     f"{max(v):.1f}")
        lut = os.path.join(C18, "seed_variance_results.json")
        if os.path.exists(lut):
            L = json.load(open(lut))
            lines.append(f"*LUT was {L['mean']:.1f} ± {L['sd']:.1f}* — sd ratio "
                         f"LUT/MLP = {L['sd'] / sd:.2f}× (needs ~2.25× to be "
                         f"distinguishable at 6 seeds)")
    return 100.0 * frac, done_all, n_fail, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=180)
    a = ap.parse_args()
    h = progress.progress_start("MLP-actor SAC control — 6 deterministic seeds (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c19.handle"), "w").write(h)
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
