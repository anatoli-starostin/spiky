"""Localize the K=128 round-3 hang: GROWTH ENGINE (build_pool) or RUNTIME (stdp episode)?

Replicates steady_state.main()'s round loop with timestamped, flushed instrumentation rather
than editing the production script. Fixed seed 0 -- the same default both hung runs used, and
their rounds 0-2 were byte-identical across launches, so the trajectory is deterministic.

The genomes are dumped BEFORE build_pool each round, so whichever stage stalls we still hold
its exact input for a standalone reproducer.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

import steady_state as S

T0 = time.time()
FIELDS = ("src_pool", "src_idx", "tgt_pool", "tgt_idx", "delay", "weight")


def log(msg):
    print(f"[{time.time()-T0:8.2f}s] {msg}", flush=True)


def dump(genomes, path, extra=None):
    d = {f"g{i}_{f}": g[f] for i, g in enumerate(genomes) for f in FIELDS}
    if extra:
        d.update(extra)
    np.savez_compressed(path, **d)
    log(f"DUMPED {len(genomes)} genomes -> {path}")


ap = argparse.ArgumentParser()
ap.add_argument("--pool", type=int, default=128)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--mature-batches", type=int, default=32)
ap.add_argument("--stdp-lr", type=float, default=0.01)
ap.add_argument("--w-max", type=float, default=30.0)
ap.add_argument("--cull", type=float, default=0.25)
ap.add_argument("--grace", type=int, default=2)
ap.add_argument("--alpha", type=float, default=0.3)
ap.add_argument("--current", type=float, default=200.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outdir", default="results")
a = ap.parse_args()

dev = "cuda"
rng = np.random.default_rng(a.seed)
X, Y, Xpool, Ypool, Xval, Yval = S.load(a.batch, a.seed, 2000)
enc = S.LatencyEncoder(Xpool)
M = max(1, int(a.cull * a.pool))
genomes = [S.seed_genome(np.random.default_rng(a.seed * 100 + i), a.w_max)
           for i in range(a.pool)]
ewma = np.full(a.pool, np.nan)
age = np.zeros(a.pool, int)
newborn = np.zeros(a.pool, bool)
log(f"K={a.pool} M={M} seed={a.seed} -- seeded {a.pool} genomes")

for rnd in range(a.rounds):
    log(f"=== ROUND {rnd} begin (newborns pending: {int(newborn.sum())})")
    Xb, Yb, _ = S.sample_batch(Xpool, Ypool, a.batch, a.seed, rnd)

    dump(genomes, os.path.join(a.outdir, f"hang_genomes_round{rnd}.npz"),
         extra=dict(newborn=newborn, age=age, rnd=np.array([rnd])))

    log(f"round {rnd}: calling build_pool ...")
    h = S.build_pool(genomes, dev, seed=1, stdp_lr=a.stdp_lr, w_max=a.w_max)
    torch.cuda.synchronize()
    log(f"round {rnd}: BUILD_POOL RETURNED  ({h['n_syn']:,} synapses)")

    if a.stdp_lr > 0 and newborn.any():
        log(f"round {rnd}: maturation, {a.mature_batches} episodes")
        for i in range(a.mature_batches):
            Xm, _, _ = S.sample_batch(Xpool, Ypool, a.batch, a.seed, rnd * 1000 + i)
            log(f"round {rnd}:   episode {i} START")
            S.run_episode(h, Xm, enc, a.current, train=True)
            torch.cuda.synchronize()
            log(f"round {rnd}:   episode {i} done")
        newborn[:] = False
        log(f"round {rnd}: maturation COMPLETE")

    log(f"round {rnd}: scoring ...")
    f, _ = S.score(h, Xb, Yb, enc, a.current)
    log(f"round {rnd}: scored, best {f.max():+.4f}")
    S.readback(h, genomes)
    log(f"round {rnd}: readback done")
    del h

    ewma = np.where(np.isnan(ewma), f, (1 - a.alpha) * ewma + a.alpha * f)
    age += 1
    eligible = np.nonzero(age > a.grace)[0]
    if eligible.size >= M:
        worst = eligible[np.argsort(ewma[eligible])[:M]]
        surv = np.setdiff1d(np.arange(a.pool), worst)
        for slot in worst:
            c1, c2 = rng.choice(surv, 2, replace=False)
            par = c1 if ewma[c1] >= ewma[c2] else c2
            genomes[slot] = S.mutate_structural(S.clone(genomes[par]), rng, a.w_max)
            ewma[slot] = ewma[par]
            age[slot] = 0
            newborn[slot] = True
        log(f"round {rnd}: culled {M}, {M} newborns created")
    log(f"=== ROUND {rnd} end")

log("ALL ROUNDS COMPLETED")
