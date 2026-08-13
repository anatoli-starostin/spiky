"""Live Slack progress bar for the full PPO-on-MJX run at solver 10/8 (issue #75).

Same rail as the SAC baseline bar (`progress.py` green-zone rendezvous, one message
edited in place by the face) but a SEPARATE handle, so it never touches the SAC bar.

On completion it flips to ✅ and posts a final summary that includes the policy's
return re-evaluated in the CPU reference env — the number that actually counts.

Usage:
    python slack_progress_ppo.py --task <BODY_TASK_ID> [--interval 45]
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.expanduser("~/work/slack-facade"))
import progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")


def read(part):
    try:
        return json.load(open(part))
    except (OSError, json.JSONDecodeError):
        return None


def bar(pct, width=16):
    f = int(round(width * pct / 100.0))
    return "█" * f + "░" * (width - f)


def fmt(sec):
    if sec is None:
        return "—"
    m = int(sec // 60)
    return f"{m//60}h{m%60:02d}m" if m >= 60 else f"{m}m{int(sec)%60:02d}s"


def cpu_reference_eval(params, episodes=20):
    """Run the CPU Walker2d-v5 reference eval in the JAX venv; -> (mean, std) or None."""
    code = (
        "import json, cross_check as C\n"
        f"net, p = C.load_policy('{params}')\n"
        f"m, s = C.eval_cpu(net, p, episodes={episodes})\n"
        "print('RESULT', json.dumps([m, s]))\n"
    )
    try:
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        out = subprocess.run([VENV, "-c", code], cwd=HERE, env=env,
                             capture_output=True, text=True, timeout=1800)
        for line in out.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[7:])
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument("--partial", default="ppo_mjx_full.json.partial")
    ap.add_argument("--params", default="ppo_policy_full.msgpack")
    ap.add_argument("--handle-file", default=os.path.join(HERE, ".slack_ppo.handle"))
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()

    h = None
    if a.reuse and os.path.exists(a.handle_file):
        cand = open(a.handle_file).read().strip()
        if cand and os.path.exists(os.path.expanduser(
                f"~/.cache/slack_facade/progress/{cand}.json")):
            h = cand
    if h is None:
        h = progress.progress_start("MJX/PPO Walker2d — GPU-parallel (#75, solver 10/8)",
                                    task=a.task, style="emoji", width=10)
    open(a.handle_file, "w").write(h)
    print(f"ppo bar handle {h} (task {a.task}) interval {a.interval}s", flush=True)

    part = os.path.join(HERE, a.partial)
    while True:
        p = read(part)
        if p is None:
            progress.progress_update(h, pct=0.0, stats="starting (JIT compiling)…")
            time.sleep(a.interval)
            continue

        pct = 100.0 * p["env_steps"] / max(p["target_env_steps"], 1)
        stats = (f"iter {p['iter']+1}/{p['iters']} · "
                 f"{p['env_steps']/1e6:.1f}M/{p['target_env_steps']/1e6:.0f}M env-steps\n"
                 f"{bar(pct)} · return ~{p['est_return_1000']:.0f} · "
                 f"{p['env_steps_per_sec']:,.0f} env-steps/s · ETA {fmt(p['eta_s'])}")

        if p.get("done"):
            ref = cpu_reference_eval(os.path.join(HERE, a.params))
            tail = (f"{p['target_env_steps']/1e6:.0f}M env-steps in "
                    f"{fmt(p['elapsed_s'])} at {p['env_steps_per_sec']:,.0f} env-steps/s · "
                    f"MJX@10/8 return ~{p['est_return_1000']:.0f}")
            if ref:
                tail += (f" · **CPU Walker2d-v5 reference: {ref[0]:.0f} ± {ref[1]:.0f}**"
                         f" ({'✅ ≥3000' if ref[0] >= 3000 else 'below the 3000 bar'})")
            else:
                tail += " · CPU reference eval unavailable"
            progress.progress_done(h, ok=True, final_text=tail)
            print("finished; posted final summary:", tail, flush=True)
            return

        progress.progress_update(h, pct=pct, stats=stats)
        print(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% iter {p['iter']+1}/{p['iters']} "
              f"ret ~{p['est_return_1000']:.0f}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
