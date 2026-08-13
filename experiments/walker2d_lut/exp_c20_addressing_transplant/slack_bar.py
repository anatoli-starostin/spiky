"""exp_c20 — live Slack bar for the addressing-transplant test (#75).

Six cells in two arms. The bar labels the ARM on every row, because the number people will
want to read off is the arm difference, and a bar that showed six anonymous cells would
invite reading arm A against seed 4's 5286.6 -- the comparison that confounds freezing with
routing. Failure is checked before progress, so a dead run cannot sit at a healthy 90%.
"""
import argparse, json, os, re, statistics, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+(\S+)\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
SEEDS = (100, 101, 102)
ARMS = (("from4", "seed 4 routing"), ("from5", "seed 5 routing (control)"))
CELLS = [(a, s) for a, _ in ARMS for s in SEEDS]
TRAIN_W = 0.9
REF = ("frozen transplanted addressing, table content relearned · 3 fresh seeds per arm · "
       "arm A = seed 4's routing, arm B = a pack seed's routing · the answer is A − B, "
       "not A vs 5286.6 (freezing costs return on its own)")


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def started():
    try:
        return "starting the transplant" in open(
            os.path.join(HERE, "run_transplant.log"), errors="replace").read()
    except OSError:
        return False


def read(arm, seed):
    ev = os.path.join(C09, f"lut_sac_c20_{arm}_s{seed}_cpueval.json")
    st = dict(arm=arm, seed=seed, eval_done=os.path.exists(ev), cpu=None)
    if st["eval_done"]:
        try:
            st["cpu"] = json.load(open(ev))["cpu_reference_mean"]
        except Exception:
            st["eval_done"] = False
    try:
        txt = open(os.path.join(HERE, f"cell_{arm}_s{seed}.log"),
                   errors="replace").read()
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
        it, tot, _ret, _cov, best = rows[-1]
        st.update(state="training", pct=100.0 * int(it) / int(tot), it=int(it),
                  tot=int(tot), best=float(best))
    else:
        st.update(state="starting", pct=0.0)
    return st


def render():
    if not started():
        return 0.0, False, 0, ("*waiting for the MLP control to finish* — queued so it "
                               "takes no GPU from exp_c19.\n`6 transplant runs` "
                               + bar(0) + "  not started")
    sts = [read(a, s) for a, s in CELLS]
    frac = sum(TRAIN_W * s["pct"] / 100.0
               + (1 - TRAIN_W) * (1.0 if s["eval_done"] else 0.0)
               for s in sts) / len(sts)
    n_fail = sum(1 for s in sts if s["state"] == "failed")
    n_eval = sum(1 for s in sts if s["eval_done"])
    n_train = sum(1 for s in sts if s["state"] == "trained" or s["eval_done"])
    done_all = n_eval == len(sts)
    running = sum(1 for s in sts if s["state"] in ("training", "starting"))

    head = ("all 6 transplant runs complete" if done_all
            else f"{n_train}/6 trained · {n_eval}/6 evaluated · {running} on GPU now")
    lines = [f"*{head}*" + (f" · *{n_fail} FAILED*" if n_fail else "")]
    for arm, lab in ARMS:
        lines.append(f"_{lab}_")
        for s in [t for t in sts if t["arm"] == arm]:
            tag = f"s{s['seed']}"
            if s["eval_done"]:
                lines.append(f"`{tag}` {bar(100)} ✅ CPU-ref *{s['cpu']:.1f}*")
            elif s["state"] == "trained":
                lines.append(f"`{tag}` {bar(100)} trained (best MJX {s['best']:.0f}) · "
                             f"eval pending…")
            elif s["state"] == "training":
                lines.append(f"`{tag}` {bar(s['pct'])} {s['pct']:3.0f}% · "
                             f"{s['it']:,}/{s['tot']:,} · best MJX {s['best']:.0f}")
            elif s["state"] == "failed":
                lines.append(f"`{tag}` {bar(0)} ❌ failed")
            else:
                lines.append(f"`{tag}` {bar(0)}  queued")

    if done_all:
        va = [t["cpu"] for t in sts if t["arm"] == "from4"]
        vb = [t["cpu"] for t in sts if t["arm"] == "from5"]
        ma, mb = statistics.mean(va), statistics.mean(vb)
        lines.append(f"\n*arm A {ma:.1f} ± {statistics.stdev(va):.1f}* vs "
                     f"*arm B {mb:.1f} ± {statistics.stdev(vb):.1f}* · "
                     f"*A − B = {ma-mb:+.1f}*")
        lines.append(f"_(seed 4 trained jointly: 5286.6 · exp_c18 pack: 4112 ± 159)_")
    return 100.0 * frac, done_all, n_fail, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=180)
    a = ap.parse_args()
    h = progress.progress_start("addressing transplant — does seed 4's routing "
                                "carry the win? (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c20.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        pct, done_all, n_fail, body = render()
        if done_all or n_fail == len(CELLS):
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
