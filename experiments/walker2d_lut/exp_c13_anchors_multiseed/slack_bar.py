"""exp_c13 — self-updating Slack bar for the 27-run multi-seed sweep (#75).

27 lines would drown a chat message, so the layout is: a header with done/running/
queued, the three RUNNING cells in full detail (nap/tph/seed, step, best MJX), and a
per-config roll-up showing each config's three seeds as they land. That keeps the whole
thing to roughly a dozen lines while still answering "where is it" at a glance.
"""
import argparse, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+([\d.]+)%\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
CONFIGS = [(n, t) for n in (6, 7, 8) for t in (32, 64, 128)]
SEEDS = (0, 1, 2)
TOTAL = len(CONFIGS) * len(SEEDS)
REF = ("exp_c12 one-seed anchors×hard best = 4880 (nap6/tph64) · "
       "target hyperplane×hard = 5147")


def bar(pct, width=12):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def read(nap, tph, seed):
    p = os.path.join(HERE, f"cell_nap{nap}_tph{tph}_s{seed}.log")
    try:
        txt = open(p, errors="replace").read()
    except OSError:
        return dict(state="queued", pct=0.0)
    d = DONE.search(txt)
    rows = ITER.findall(txt)
    if d:
        return dict(state="done", pct=100.0, best=float(d.group(1)),
                    cov=float(rows[-1][3]) if rows else 0.0)
    # Check for failure BEFORE reporting progress: a run that printed 9,000 iterations
    # and then died must not keep showing as a healthy 90%.
    if re.search(r"Traceback|CUDA out of memory|Killed", txt):
        return dict(state="failed", pct=0.0)
    if rows:
        it, tot, ret, cov, best = rows[-1]
        return dict(state="running", pct=100.0 * int(it) / int(tot), it=int(it),
                    tot=int(tot), best=float(best), cov=float(cov))
    return dict(state="starting", pct=0.0)


def render():
    st = {(n, t, s): read(n, t, s) for n, t in CONFIGS for s in SEEDS}
    n_done = sum(1 for v in st.values() if v["state"] == "done")
    n_fail = sum(1 for v in st.values() if v["state"] == "failed")
    running = [(k, v) for k, v in sorted(st.items())
               if v["state"] in ("running", "starting")]
    n_queued = TOTAL - n_done - n_fail - len(running)
    pct = sum(v["pct"] for v in st.values()) / TOTAL

    lines = [f"*{n_done}/{TOTAL} done* · {len(running)} running · {n_queued} queued"
             + (f" · *{n_fail} FAILED*" if n_fail else "")]
    if running:
        lines.append("")
        for (nap, tph, seed), v in running:
            lab = f"nap{nap}/tph{tph} s{seed}"
            if v["state"] == "starting":
                lines.append(f"`{lab:<15}` {bar(0)}  starting…")
            else:
                lines.append(f"`{lab:<15}` {bar(v['pct'])} {v['pct']:3.0f}% · "
                             f"{v['it']:,}/{v['tot']:,} · best MJX {v['best']:.0f}")
    lines.append("")
    for nap, tph in CONFIGS:
        cells = []
        for s in SEEDS:
            v = st[(nap, tph, s)]
            cells.append(f"s{s} {v['best']:.0f}" if v["state"] == "done"
                         else (f"s{s} ✗" if v["state"] == "failed"
                               else (f"s{s} ·" if v["state"] == "queued" else f"s{s} …")))
        lines.append(f"`nap{nap}/tph{tph:<3}` " + " · ".join(cells))
    return pct, n_done, n_fail, len(running), "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=180)
    a = ap.parse_args()
    h = progress.progress_start("anchors×hard capacity sweep — 3 seeds (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c13.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        pct, n_done, n_fail, n_run, body = render()
        stats = (f"_{REF}_\n_MJX proxy shown; the CPU-reference table follows_\n"
                 + body)
        if n_done + n_fail == TOTAL:
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=(n_fail == 0),
                final_text=(f"all {TOTAL} runs complete"
                            + (f" ({n_fail} FAILED)" if n_fail else "") + "\n" + body +
                            f"\n_CPU-reference multi-seed table follows in-thread_\n"
                            f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% "
              f"({n_done} done, {n_run} running)", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
