"""Live Slack progress bar for the exp_c01 SAC runs (issue #75).

Rides the established façade rail: this writes green-zone records via
`~/work/slack-facade/progress.py` (no network, no approval from inside the cage)
and the face's reaper posts ONE Slack message and edits it in place.

One combined bar for all three seeds: the top-level bar is aggregate progress,
and `stats` carries a per-seed line each.

Data source is `run_seed<N>/progress.json` (written by the trainer's callback) with
`train_seed<N>.log` used only to detect completion. Runs detached; see launch below.

Usage:
    python slack_progress.py --task <BODY_TASK_ID> [--interval 180]
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Seeds 1 and 2 were stopped once the GPU-parallel MJX/PPO track became the main
# path; seed 0 alone is the pinned SAC reference. Their logs/checkpoints are kept.
SEEDS = (0,)
TOTAL = 1_000_000
EXIT_RE = re.compile(r"=== EXIT=(\d+)")


def read_seed(seed):
    """-> dict(step, ret, std, fps, eta_s, done, exit_code) for one run."""
    out = dict(step=0, ret=None, std=None, fps=None, eta_s=None,
               done=False, exit_code=None)
    pj = os.path.join(HERE, f"run_seed{seed}", "progress.json")
    if os.path.exists(pj):
        try:
            rows = json.load(open(pj))
            if rows:
                r = rows[-1]
                out.update(step=r["step"], ret=r["mean_return"], std=r["std_return"],
                           fps=r["fps"], eta_s=r["eta_s"])
        except (json.JSONDecodeError, KeyError, OSError):
            pass                      # mid-write; keep the previous tick's view
    log = os.path.join(HERE, f"train_seed{seed}.log")
    if os.path.exists(log):
        tail = open(log, errors="replace").read()[-4000:]
        m = EXIT_RE.search(tail)
        if m:
            out["done"] = True
            out["exit_code"] = int(m.group(1))
        fin = re.search(r"\[FINAL\] deterministic 100-episode eval: "
                        r"([-\d.]+) \+/- ([-\d.]+)", tail)
        if fin:
            out["final_mean"] = float(fin.group(1))
            out["final_std"] = float(fin.group(2))
    return out


def minibar(pct, width=10):
    filled = int(round(width * pct / 100.0))
    return "█" * filled + "░" * (width - filled)


def fmt_eta(sec):
    if sec is None:
        return "—"
    m = int(sec // 60)
    return f"{m//60}h{m%60:02d}m" if m >= 60 else f"{m}m"


def build(states):
    """-> (aggregate_pct, stats_block)"""
    agg = 100.0 * sum(s["step"] for s in states.values()) / (TOTAL * len(SEEDS))
    what = (f"seed {SEEDS[0]} · {TOTAL/1e6:g}M steps" if len(SEEDS) == 1
            else f"{len(SEEDS)} seeds × {TOTAL/1e6:g}M steps")
    lines = [f"{what} · SAC · Walker2d-v5"]
    for seed in SEEDS:
        s = states[seed]
        pct = 100.0 * s["step"] / TOTAL
        if s["done"]:
            fm = s.get("final_mean")
            tail = (f"final {fm:.0f}" if fm is not None else
                    f"exit {s['exit_code']}")
            lines.append(f"`seed {seed}` {minibar(100)} 100% · ✅ {tail}")
        else:
            ret = f"{s['ret']:.0f}" if s["ret"] is not None else "—"
            fps = f"{s['fps']:.0f}" if s["fps"] is not None else "—"
            lines.append(f"`seed {seed}` {minibar(pct)} {pct:3.0f}% · "
                         f"{s['step']:,}/{TOTAL:,} · ret {ret} · "
                         f"{fps} fps · ETA {fmt_eta(s['eta_s'])}")
    return agg, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="BODY_TASK id to post under")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--handle-file", default=os.path.join(HERE, ".slack_progress.handle"))
    ap.add_argument("--reuse", action="store_true",
                    help="keep editing the existing bar instead of posting a new one")
    a = ap.parse_args()

    # Reuse the existing bar's handle if one is on disk, so a restart of this
    # poller keeps editing the SAME Slack message rather than posting a new one.
    h = None
    if a.reuse and os.path.exists(a.handle_file):
        cand = open(a.handle_file).read().strip()
        if cand and os.path.exists(os.path.expanduser(
                f"~/.cache/slack_facade/progress/{cand}.json")):
            h = cand
            print(f"reusing existing bar handle {h}", flush=True)
    if h is None:
        h = progress.progress_start("Walker2d-v5 SAC baseline (#75)", task=a.task,
                                    style="emoji", width=10)
    open(a.handle_file, "w").write(h)
    print(f"progress handle {h} (task {a.task}), interval {a.interval}s", flush=True)

    while True:
        states = {s: read_seed(s) for s in SEEDS}
        agg, stats = build(states)
        if all(st["done"] for st in states.values()):
            finals = [st.get("final_mean") for st in states.values()]
            got = [f for f in finals if f is not None]
            if got:
                mean = sum(got) / len(got)
                solved = "✅ SOLVED (≥3000)" if mean >= 3000 else "below the 3000 bar"
                summary = (f"3 seeds done · per-seed 100-ep eval "
                           + ", ".join(f"{f:.0f}" for f in got)
                           + f" · mean {mean:.0f} — {solved}")
            else:
                summary = stats
            progress.progress_done(h, ok=True, final_text=summary)
            print("all seeds finished; posted final summary", flush=True)
            return
        progress.progress_update(h, pct=agg, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] agg {agg:.1f}% | "
              + " | ".join(f"s{s}:{states[s]['step']:,}" for s in SEEDS), flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
