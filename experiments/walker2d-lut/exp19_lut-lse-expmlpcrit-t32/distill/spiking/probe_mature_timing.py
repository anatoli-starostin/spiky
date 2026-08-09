"""Is K=128 maturation SLOW-but-linear, or does it HANG at some episode?

Both long runs stalled with GPU pinned at 100% and memory flat: K=32 at round 136 (after 135
healthy ~4s rounds) and K=128 at round 3 (its first maturation round). Round 3 is the first
round with newborns, so stdp_batches -- 32 consecutive train episodes -- is the suspect.

Time consecutive train episodes one at a time. Linear growth => merely slow; a single
episode that never returns => a genuine hang, and we learn WHICH episode index.
"""
import argparse
import sys
import time

import numpy as np
import torch

import steady_state as S

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=128)
ap.add_argument("--b", type=int, default=64)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--genomes", default=None,
                help="load genomes from a checkpoint/dump npz instead of seeding")
a = ap.parse_args()

X, Y, Xpool, Ypool, _, _ = S.load(a.b, 0, 256)
enc = S.LatencyEncoder(Xpool)
if a.genomes:
    F = ("src_pool", "src_idx", "tgt_pool", "tgt_idx", "delay", "weight")
    z = np.load(a.genomes, allow_pickle=False)
    K = int(z["n_genomes"][0]) if "n_genomes" in z.files else a.k
    genomes = [{f: z[f"g{i}_{f}"] for f in F} for i in range(K)]
    a.k = K
    print(f"loaded {K} genomes from {a.genomes}", flush=True)
else:
    genomes = [S.seed_genome(np.random.default_rng(i), 30.0) for i in range(a.k)]
t0 = time.time()
h = S.build_pool(genomes, "cuda", seed=1, stdp_lr=0.01, w_max=30.0)
print(f"build K={a.k}: {h['n_syn']:,} synapses in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
S.run_episode(h, S.sample_batch(Xpool, Ypool, a.b, 0, 0)[0], enc, 200.0, train=False)
torch.cuda.synchronize()
print(f"inference episode: {time.time()-t0:.2f}s", flush=True)

for i in range(a.n):
    Xb, _, _ = S.sample_batch(Xpool, Ypool, a.b, 0, i)
    t0 = time.time()
    S.run_episode(h, Xb, enc, 200.0, train=True)
    torch.cuda.synchronize()
    print(f"  train episode {i}: {time.time()-t0:7.2f}s", flush=True)
print("ALL EPISODES COMPLETED")
