"""exp_c09 — timestamped Slack bar for LUT-SAC, showing progress AND best-so-far
against the three anchors (#75)."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHORS = "anchors · PPO-scratch 4407 · SAC 5277 · distillation ceiling 5512"


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def fmt(sec):
    m = int((sec or 0) // 60)
    return f"{m//60}h{m%60:02d}m" if m >= 60 else f"{m}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--partial", required=True)
    ap.add_argument("--label", default="LUT-SAC from scratch (#75)")
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    h = progress.progress_start(a.label, task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c09.handle"), "w").write(h)
    print(f"bar {h}", flush=True)
    p = os.path.join(HERE, a.partial)

    while True:
        try:
            d = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            progress.progress_update(h, pct=0.0, stats="starting (JIT compiling)…")
            time.sleep(a.interval)
            continue
        pct = 100.0 * d["iter"] / max(d["iters"], 1)
        stats = (f"{ANCHORS}\n"
                 f"{bar(pct)} iter {d['iter']:,}/{d['iters']:,} · "
                 f"{d['env_steps']/1e6:.2f}M env-steps · "
                 f"MJX ret {d['mjx_return']:.0f} · **best {d['best']:.0f}** · "
                 f"row-coverage {d['row_coverage']*100:.1f}% · ETA {fmt(d.get('eta_s'))}")
        if d.get("done"):
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=True,
                final_text=(f"training complete — best MJX {d['best']:.0f}, "
                            f"row-coverage {d['row_coverage']*100:.1f}%\n"
                            f"_MJX return is a horizon-1000 proxy; the CPU-reference "
                            f"100-episode number follows in-thread._\n"
                            f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% best {d['best']:.0f}",
              flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
