"""Replay the WHOLE of round 6 on the checkpointed genomes, stage by stage.

Build alone is fine (3/3) and 32 maturation episodes alone are fine (32/32) on these exact
genomes, yet the supervised run failed to complete round 6 three times. So the stall is in a
stage those two probes never reached: score(), readback(), or teardown.
"""
import time

import numpy as np
import torch

import steady_state as S

F = ("src_pool", "src_idx", "tgt_pool", "tgt_idx", "delay", "weight")
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.2f}s] {m}", flush=True)


z = np.load("results/hang_repro_round6_k128.npz", allow_pickle=False)
K = int(z["n_genomes"][0])
genomes = [{f: z[f"g{i}_{f}"] for f in F} for i in range(K)]
log(f"loaded {K} genomes")

X, Y, Xpool, Ypool, _, _ = S.load(64, 0, 2000)
enc = S.LatencyEncoder(Xpool)
Xb, Yb, _ = S.sample_batch(Xpool, Ypool, 64, 0, 6)
log("data ready")

h = S.build_pool(genomes, "cuda", seed=1, stdp_lr=0.01, w_max=30.0)
torch.cuda.synchronize()
log(f"BUILD done ({h['n_syn']:,} synapses)")

for i in range(32):
    Xm, _, _ = S.sample_batch(Xpool, Ypool, 64, 0, 6 * 1000 + i)
    S.run_episode(h, Xm, enc, 200.0, train=True)
torch.cuda.synchronize()
log("MATURATION done (32 episodes)")

f, _ = S.score(h, Xb, Yb, enc, 200.0)
log(f"SCORE done (best {f.max():+.4f})")

n = S.readback(h, genomes)
log(f"READBACK done ({n:,} weights pulled)")

del h
torch.cuda.empty_cache()
log("TEARDOWN done -- ROUND 6 FULLY REPLAYED, no stall")
