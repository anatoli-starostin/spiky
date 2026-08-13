"""Fresh Slack bar for Phase 4 (#75) — timestamped, so a finished bar reads as finished.

Fixes the earlier confusion two ways: the final text carries a UTC timestamp, and it
states explicitly that the bar has stopped on purpose.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def fmt(sec):
    if sec is None:
        return "—"
    m = int(sec // 60)
    return f"{m//60}h{m%60:02d}m" if m >= 60 else f"{m}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--partial", default="ppo_lut_scratch.json.partial")
    a = ap.parse_args()

    h = progress.progress_start(
        "Phase 4 — LUT from scratch by backprop (#75)", task=a.task,
        style="emoji", width=10)
    open(os.path.join(HERE, ".slack_p4.handle"), "w").write(h)
    print(f"phase4 bar {h} interval {a.interval}s", flush=True)
    p = os.path.join(HERE, a.partial)

    while True:
        try:
            d = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            progress.progress_update(h, pct=0.0, stats="starting (JIT compiling)…")
            time.sleep(a.interval)
            continue
        pct = 100.0 * d["env_steps"] / max(d["target_env_steps"], 1)
        stats = (f"gradients verified vs torch (worst rel 2.5e-06) · "
                 f"random init, no teacher\n"
                 f"{bar(pct)} iter {d['iter']+1}/{d['iters']} · "
                 f"{d['env_steps']/1e6:.1f}M/{d['target_env_steps']/1e6:.0f}M env-steps "
                 f"· proxy ~{d['proxy_return']:.0f} · "
                 f"{d['env_steps_per_sec']:,.0f} sps · ETA {fmt(d['eta_s'])}")
        if d.get("done"):
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=True,
                final_text=(f"training complete — {d['env_steps']/1e6:.1f}M env-steps, "
                            f"proxy ~{d['proxy_return']:.0f}\n"
                            f"_CPU-reference 100-episode eval follows in-thread; the "
                            f"proxy is not comparable to the 3000 bar._\n"
                            f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished; posted final", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% iter {d['iter']+1}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
