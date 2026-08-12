"""exp012 full run: launch N seeds of tiny_evolve concurrently under ONE Slack progress bar.

WHY A DRIVER OWNS THE BAR. progress.py's record is a single file per handle, and a handle is
single-writer by design. Eight seeds each calling progress_update on the same handle would
race and the bar would jump backwards; eight seeds each calling progress_start would post
eight bars into the thread. So the seeds write a one-line heartbeat file per round
(`<tag>.progress`: "round total best_heldout") and this driver polls them, sums, and makes the
only Slack calls in the whole run.

Resumable: pass --resume and each seed picks up from its own ck_<tag>.npz.

    python tiny_run_full.py --seeds 8 --rounds 1500 --w-max 60 --out-dir ../exp012_.../full_run
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/astarostin/work/slack-facade")
import progress                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/astarostin/projects/spiky/.venv/bin/python"


def read_beats(out_dir, tags):
    done = tot = 0
    best = []
    for t in tags:
        p = os.path.join(out_dir, f"{t}.progress")
        try:
            with open(p) as f:
                r, n, b = f.read().split()
            done += int(r)
            tot += int(n)
            b = float(b)
            if math.isfinite(b):
                best.append(b)
        except (OSError, ValueError):
            pass
    return done, tot, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--w-max", type=float, default=60.0)
    ap.add_argument("--cull", type=int, default=None,
                    help="members replaced per round. Default scales with the pool to hold "
                         "the validated 12.5 %% turnover (pool 64 / cull 8), so a bigger pool "
                         "buys diversity rather than 8x fewer generations.")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", default=None, help="BODY_TASK id, so the bar lands in its thread")
    ap.add_argument("--label", default="exp012 full run")
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--runner", default="tiny_evolve.py",
                    help="which evolve script to launch (tiny_evolve.py fixed-size, "
                         "tiny_grow_evolve.py growable)")
    ap.add_argument("--extra", default="", help="extra flags passed through to the runner")
    ap.add_argument("--tag-prefix", default="s")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    tags = [f"{a.tag_prefix}{s}" for s in range(a.seeds)]
    total_rounds = a.seeds * a.rounds

    procs = []
    for s in range(a.seeds):
        cmd = [PY, "-u", os.path.join(HERE, a.runner),
               "--seed", str(s), "--rounds", str(a.rounds), "--pool", str(a.pool),
               "--batch", str(a.batch),
               # --w-max belongs to the ABSOLUTE-weight runner only; the growable runner
               # carries normalised weights and takes --sigma/--gain instead
               *([] if "grow" in a.runner else ["--w-max", str(a.w_max)]),
               "--ckpt-every", str(a.ckpt_every),
               "--cull", str(a.cull if a.cull is not None else max(1, a.pool // 8)),
               "--tag", f"{a.tag_prefix}{s}", "--out-dir", a.out_dir]
        if a.crossover:
            cmd.append("--crossover")
        if a.resume:
            cmd.append("--resume")
        if a.extra:
            cmd += a.extra.split()
        log = open(os.path.join(a.out_dir, f"{a.tag_prefix}{s}.log"), "a" if a.resume else "w")
        env = dict(os.environ, TRITON_CACHE_DIR="/tmp/triton_cache",
                   MPLCONFIGDIR="/tmp/mpl")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))

    h = progress.progress_start(a.label, task=a.task) if a.task else \
        progress.progress_start(a.label)
    print(f"bar handle {h}", flush=True)
    t0 = time.time()
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(a.poll)
            done, _, best = read_beats(a.out_dir, tags)
            el = time.time() - t0
            eta = (el / done * (total_rounds - done)) if done else 0
            alive = sum(p.poll() is None for p in procs)
            stats = (f"{alive}/{a.seeds} seeds running · "
                     f"best held-out MSE {min(best):.2f}" if best else f"{alive} seeds running")
            if best:
                stats += f" (chance ~34.2) · eta ~{eta / 60:.0f}m"
            progress.progress_update(h, step=done, total=total_rounds, stats=stats)
    finally:
        codes = [p.wait() for p in procs]

    done, _, best = read_beats(a.out_dir, tags)
    ok = all(c == 0 for c in codes)
    finals = {}
    for s in range(a.seeds):
        p = os.path.join(a.out_dir, f"{a.tag_prefix}{s}_final.json")
        if os.path.exists(p):
            finals[s] = json.load(open(p))["best"]["heldout_mse"]
    txt = (f"{len(finals)}/{a.seeds} seeds finished · best held-out MSE "
           f"{min(finals.values()):.2f} vs chance ~34.2" if finals else "no seed finished")
    progress.progress_update(h, step=total_rounds if ok else done, total=total_rounds,
                             stats=txt)
    progress.progress_done(h, ok=ok, final_text=txt)
    print(f"exit codes {codes}; {txt}; {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
