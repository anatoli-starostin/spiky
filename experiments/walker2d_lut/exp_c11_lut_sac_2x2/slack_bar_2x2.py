"""exp_c11 — one consolidated, timestamped Slack bar for the four LUT-SAC 2x2 cells.

Reads the per-cell training logs directly, so it needs no cooperation from the runs and
picks up cells that have not started yet. Overall progress is cells-completed plus the
in-flight cell's own fraction.
"""
import argparse, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHORS = "anchors · PPO-scratch 4407 · SAC 5277 · distill-smooth 5512 · distill-hard 3869"
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+([\d,]+)\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+([\d.]+)%\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)

CELLS = [("hyperplane", "hard"), ("hyperplane", "hybrid_smooth"),
         ("anchors", "hard"), ("anchors", "hybrid_smooth")]


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def read(addr, mode):
    p = os.path.join(HERE, f"cell_c11_{addr}_{mode}.log")
    try:
        txt = open(p, errors="replace").read()
    except OSError:
        return dict(state="queued", pct=0.0)
    d = DONE.search(txt)
    rows = ITER.findall(txt)
    if d:
        cov = float(rows[-1][4]) if rows else 0.0
        return dict(state="done", pct=100.0, best=float(d.group(1)), cov=cov)
    if rows:
        it, tot, steps, ret, cov, best = rows[-1]
        return dict(state="running", pct=100.0 * int(it) / int(tot), it=int(it),
                    tot=int(tot), best=float(best), cov=float(cov))
    if "Traceback" in txt:
        return dict(state="failed", pct=0.0)
    return dict(state="starting", pct=0.0)


def build():
    lines, pcts, n_done = [], [], 0
    for addr, mode in CELLS:
        s = read(addr, mode)
        pcts.append(s["pct"])
        label = f"{addr[:4]}·{'hard ' if mode == 'hard' else 'smooth'}"
        if s["state"] == "done":
            n_done += 1
            lines.append(f"`{label}` {bar(100)} 100% · ✅ best MJX {s['best']:.0f} "
                         f"· cov {s['cov']:.0f}%")
        elif s["state"] == "running":
            lines.append(f"`{label}` {bar(s['pct'])} {s['pct']:3.0f}% · "
                         f"iter {s['it']:,}/{s['tot']:,} · best MJX {s['best']:.0f} "
                         f"· cov {s['cov']:.0f}%")
        elif s["state"] == "failed":
            lines.append(f"`{label}` {bar(0)}   — · ❌ failed")
        else:
            lines.append(f"`{label}` {bar(0)}   — · queued")
    head = (f"{n_done}/4 cells complete · LUT-SAC ratio 0.5, nap6/tph32, 28k params\n"
            f"_{ANCHORS}_\n_MJX return is the horizon-1000 proxy; CPU-reference "
            f"evals follow in-thread_")
    return sum(pcts) / len(CELLS), head + "\n" + "\n".join(lines), n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=90)
    a = ap.parse_args()
    h = progress.progress_start("LUT-SAC real-training 2×2 (#75)", task=a.task,
                                style="emoji", width=10)
    open(os.path.join(HERE, ".slack_2x2.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        pct, stats, n_done = build()
        if n_done == len(CELLS):
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=True,
                final_text=(f"all 4 cells complete\n{stats.split(chr(10), 3)[3]}\n"
                            f"_CPU-reference 100-episode evals + the 2×2 table follow "
                            f"in-thread_\n_finished {stamp} — this bar has stopped on "
                            f"purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% ({n_done}/4 done)", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
