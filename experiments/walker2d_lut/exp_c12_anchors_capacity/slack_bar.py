"""exp_c12 — timestamped Slack bar for the anchors x hard capacity sweep (#75)."""
import argparse, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]\s+steps\s+[\d,]+\s+\|\s+MJX ret\s+([-\d.]+)"
                  r"\s+\|\s+row-cov\s+([\d.]+)%\s+\|\s+best\s+([-\d.]+)", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
CELLS = [(7, 64), (8, 128), (6, 128), (8, 32), (6, 64), (7, 32), (7, 128), (8, 64)]
REF = "anchors×hard baseline nap6/tph32 = 4302 · target hyperplane×hard = 5147 (28k params)"


def bar(pct, width=12):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def read(nap, tph):
    p = os.path.join(HERE, f"cell_nap{nap}_tph{tph}.log")
    try:
        txt = open(p, errors="replace").read()
    except OSError:
        return dict(state="queued", pct=0.0)
    d = DONE.search(txt)
    rows = ITER.findall(txt)
    if d:
        return dict(state="done", pct=100.0, best=float(d.group(1)),
                    cov=float(rows[-1][3]) if rows else 0.0)
    if rows:
        it, tot, ret, cov, best = rows[-1]
        return dict(state="running", pct=100.0 * int(it) / int(tot), it=int(it),
                    tot=int(tot), best=float(best), cov=float(cov))
    if "Traceback" in txt:
        return dict(state="failed", pct=0.0)
    return dict(state="starting", pct=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=120)
    a = ap.parse_args()
    h = progress.progress_start("anchors×hard capacity sweep (#75)", task=a.task,
                                style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c12.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    while True:
        lines, pcts, n_done = [], [], 0
        for nap, tph in CELLS:
            s = read(nap, tph)
            pcts.append(s["pct"])
            lab = f"nap{nap}/tph{tph}"
            if s["state"] == "done":
                n_done += 1
                lines.append(f"`{lab:<12}` {bar(100)} ✅ best MJX {s['best']:.0f} "
                             f"· cov {s['cov']:.0f}%")
            elif s["state"] == "running":
                lines.append(f"`{lab:<12}` {bar(s['pct'])} {s['pct']:3.0f}% · "
                             f"{s['it']:,}/{s['tot']:,} · best {s['best']:.0f}")
            elif s["state"] == "failed":
                lines.append(f"`{lab:<12}` {bar(0)} ❌ failed")
            else:
                lines.append(f"`{lab:<12}` {bar(0)}  queued")
        pct = sum(pcts) / len(CELLS)
        stats = (f"{n_done}/8 cells · _{REF}_\n"
                 f"_MJX proxy shown; CPU-reference evals follow_\n" + "\n".join(lines))
        if n_done == len(CELLS):
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=True,
                final_text=(f"all 8 cells complete\n" + "\n".join(lines) +
                            f"\n_CPU-reference table follows in-thread_\n"
                            f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% ({n_done}/8)", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
