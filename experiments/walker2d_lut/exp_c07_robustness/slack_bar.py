"""exp_c07 — timestamped Slack bar for the robustness sweeps (#75)."""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^\s+(\S+)\s+(\w+)\s+x([\d.]+)\s+->", re.M)
TOTAL_PER_POLICY = 18
POLICIES = 4


def bar(pct, width=14):
    f = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "█" * f + "░" * (width - f)


def count(path):
    try:
        return len(ROW.findall(open(path, errors="replace").read()))
    except OSError:
        return 0


def done(path):
    try:
        return "wrote results_" in open(path, errors="replace").read()
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=45)
    a = ap.parse_args()
    h = progress.progress_start("Phase 5 — zero-shot robustness sweep (#75)",
                                task=a.task, style="emoji", width=10)
    open(os.path.join(HERE, ".slack_c07.handle"), "w").write(h)
    print(f"bar {h}", flush=True)

    t_log = os.path.join(HERE, "run_torch.log")
    j_log = os.path.join(HERE, "run_jax.log")
    while True:
        nt, nj = count(t_log), count(j_log)
        tot = TOTAL_PER_POLICY * POLICIES
        pct = 100.0 * (nt + nj) / tot
        stats = (f"4 frozen policies × 18 perturbed envs × 100 deterministic episodes\n"
                 f"{bar(pct)} {nt + nj}/{tot} cells · "
                 f"torch(LUT-distilled, SAC) {nt}/36 · jax(PPO, LUT-scratch) {nj}/36")
        if done(t_log) and done(j_log):
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            progress.progress_done(
                h, ok=True,
                final_text=(f"sweep complete — {nt + nj} cells\n"
                            f"_curves + verdict posted in-thread_\n"
                            f"_finished {stamp} — this bar has stopped on purpose._"))
            print("finished", flush=True)
            return
        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% ({nt}+{nj})", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
