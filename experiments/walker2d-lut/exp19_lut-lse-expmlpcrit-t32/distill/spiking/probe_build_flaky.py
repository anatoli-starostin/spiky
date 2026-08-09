"""Build-only reproducer: how often does build_pool misbehave on IDENTICAL input?

Fixed seed, identical genomes every attempt, one fresh process per attempt. Three outcomes
seen so far at K=128: clean build, ValueError from the create_forward_groups error counter,
and an outright hang. Same input producing different outcomes across processes is the
signature of a RACE in the growth kernels, not of a bad genome.

    python probe_build_flaky.py --k 128 --tries 8
"""
import argparse
import subprocess
import sys

import numpy as np


def build_once(K, stdp_lr, w_max=30.0):
    import torch
    import steady_state as S
    genomes = [S.seed_genome(np.random.default_rng(i), w_max) for i in range(K)]
    h = S.build_pool(genomes, "cuda", seed=1, stdp_lr=stdp_lr, w_max=w_max)
    torch.cuda.synchronize()
    print(f"BUILD-OK {h['n_syn']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--tries", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--stdp-lr", type=float, default=0.01)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        build_once(a.k, a.stdp_lr)
        sys.exit(0)

    tally = {"ok": 0, "error": 0, "hang": 0}
    for t in range(a.tries):
        try:
            r = subprocess.run(
                [sys.executable, __file__, "--child", "--k", str(a.k),
                 "--stdp-lr", str(a.stdp_lr)],
                capture_output=True, text=True, timeout=a.timeout)
            txt = r.stdout + r.stderr
            if "BUILD-OK" in txt:
                tally["ok"] += 1
                v = "ok"
            else:
                tally["error"] += 1
                e = [l.strip() for l in txt.splitlines() if "Error" in l]
                v = "ERROR: " + (e[-1][:70] if e else f"rc={r.returncode}")
        except subprocess.TimeoutExpired:
            tally["hang"] += 1
            v = f"HANG (no return in {a.timeout}s)"
        print(f"  K={a.k} attempt {t}: {v}", flush=True)
    n = a.tries
    print(f"\n  K={a.k}: ok {tally['ok']}/{n}, error {tally['error']}/{n}, "
          f"hang {tally['hang']}/{n}")
