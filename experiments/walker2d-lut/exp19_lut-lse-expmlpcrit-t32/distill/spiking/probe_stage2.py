"""Bisect the stage-two training crash (spnet_runtime.cu:1030) on the REAL steady_state path.

The group_size fix cleared the BUILD crash, and a 40-meta plastic+frozen bank trains fine at
small scale in test_multimeta -- so meta layout is not the trigger. Remaining suspects are
pool size K and batch size B. One subprocess per cell: a CUDA fault poisons the context.
"""
import argparse
import subprocess
import sys

import numpy as np
import torch


def one(K, B, stdp_lr):
    import steady_state as S
    X, Y, Xpool, Ypool, Xval, Yval = S.load(B, 0, 256)
    enc = S.LatencyEncoder(Xpool)
    genomes = [S.seed_genome(np.random.default_rng(i), 30.0) for i in range(K)]
    h = S.build_pool(genomes, "cuda", seed=1, stdp_lr=stdp_lr, w_max=30.0)
    print(f"BUILT K={K} synapses={h['n_syn']:,}")
    Xb, _, _ = S.sample_batch(Xpool, Ypool, B, 0, 0)
    S.run_episode(h, Xb, enc, 200.0, train=False)
    print("INFER-OK")
    S.run_episode(h, Xb, enc, 200.0, train=True)
    torch.cuda.synchronize()
    print("TRAIN-OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--b", type=int, default=256)
    ap.add_argument("--stdp-lr", type=float, default=0.1)
    a = ap.parse_args()
    if a.k:
        one(a.k, a.b, a.stdp_lr)
    else:
        for K, B in ((1, 256), (2, 256), (4, 256), (8, 256),
                     (8, 64), (8, 16), (1, 512), (4, 64)):
            r = subprocess.run([sys.executable, __file__, "--k", str(K), "--b", str(B),
                                "--stdp-lr", str(a.stdp_lr)],
                               capture_output=True, text=True)
            txt = r.stdout + r.stderr
            stage = ("TRAIN-OK" if "TRAIN-OK" in txt else
                     "crash in TRAIN" if "INFER-OK" in txt else
                     "crash in INFER" if "BUILT" in txt else "crash in BUILD")
            err = [ln.strip() for ln in txt.splitlines() if "Error" in ln]
            nsyn = [ln for ln in txt.splitlines() if ln.startswith("BUILT")]
            detail = "" if stage == "TRAIN-OK" else "  " + (
                err[-1].split("error")[-1].strip()[:64] if err else f"rc={r.returncode}")
            print(f"  K={K:2d} B={B:4d}  {nsyn[0].split('synapses=')[1] if nsyn else '?':>10s} "
                  f"-> {stage}{detail}", flush=True)
