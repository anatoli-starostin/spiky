"""exp_c14 — self-updating Slack bar for the 3-seed hyperplane reference (#75)."""
import argparse, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+([\d.]+)%\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
SEEDS = (0, 1, 2)
REF = ("single-seed hyperplane×hard = 5147 · best anchors 3-seed = 4678 ± 474 "
       "(nap7/tph64) · anchors seed-sd ran 131–1850")


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def read(seed):
    p = os.path.join(HERE, f"cell_hyperplane_hard_s{seed}.log")
    try:
        txt = open(p, errors="replace").read()
    except OSError:
        return dict(state="queued", pct=0.0)
    d = DONE.search(txt)
    rows = ITER.findall(txt)
    if d:
        return dict(state="done", pct=100.0, best=float(d.group(1)),
                    cov=float(rows[-1][3]) if rows else 0.0)
    # failure checked BEFORE progress: a run that died at 9,000 iters must not keep
    # showing as a healthy 90%.
    if re.search(r"Traceback|CUDA out of memory|Killed", txt):
        return dict(state="failed", pct=0.0)
    if rows:
        it, tot, ret, cov, best = rows[-1]
        return dict(state="running", pct=100.0 * int(it) / int(tot), it=int(it),
                    tot=int(tot), best=float(best), cov=float(cov))
    return dict(state="starting", pct=0.0)


def render():
    st = {s: read(s) for s in SEEDS}
    n_done = sum(1 for v in st.values() if v["state"] == "done")
    n_fail = sum(1 for v in st.values() if v["state"] == "failed")
    pct = sum(v["pct"] for v in st.values()) / len(SEEDS)
    lines = [f"*{n_done}/3 done*" + (f" · *{n_fail} FAILED*" if n_fail else "")]
    for s in SEEDS:
        v = st[s]
        lab = f"hyperplane×hard s{s}"
        if v["state"] == "done":
            lines.append(f"`{lab}` {bar(100)} ✅ best MJX {v['best']:.0f} "
                         f"· cov {v['cov']:.0f}%")
        elif v["state"] == "running":
            lines.append(f"`{lab}` {bar(v['pct'])} {v['pct']:3.0f}% · "
                         f"{v['it']:,}/{v['tot']:,} · best {v['best']:.0f}")
        elif v["state"] == "failed":
            lines.append(f"`{lab}` {bar(0)} ❌ failed")
        else:
            lines.append(f"`{lab}` {bar(0)}  starting…")
    return pct, n_done, n_fail, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=180)
    a = ap.parse_args()
    h = progress.progress_start("hyperplane×hard reference — 3 seeds (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c14.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        pct, n_done, n_fail, body = render()
        stats = (f"_{REF}_\n_MJX proxy shown; CPU-reference evals follow_\n" + body)
        if n_done + n_fail == len(SEEDS):
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=(n_fail == 0),
                final_text=("all 3 seeds complete"
                            + (f" ({n_fail} FAILED)" if n_fail else "") + "\n" + body +
                            f"\n_CPU-reference comparison follows in-thread_\n"
                            f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% ({n_done}/3)", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
