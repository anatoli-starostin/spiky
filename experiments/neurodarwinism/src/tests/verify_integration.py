"""Verify the vectorised aligner now living in the library: correctness + build timing.

Compares the stock _grow_explicit(weights=) path (now vectorised) against
es_harness.group_aligned_weights on the same chunk, and times a K=128 build.
"""
# tests live one level below src/; make the sibling modules importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time

import numpy as np
import torch

import steady_state as S
from harness import group_aligned_weights

# ---- 1. the two implementations must agree edge-for-edge on a real multi-meta chunk
from spiky.util.synapse_growth import SynapseGrowthEngine

for K in (2,):
    genomes = [S.seed_genome(np.random.default_rng(i), 30.0) for i in range(K)]
    t0 = time.time()
    h = S.build_pool(genomes, "cuda", seed=1, stdp_lr=0.01, w_max=30.0)
    torch.cuda.synchronize()
    print(f"K={K}: build_pool via STOCK path {time.time() - t0:.1f}s, "
          f"{h['n_syn']:,} synapses")

    ids = h["ids"]
    all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device="cuda")
    n = h["spnet"].count_synapses(all_ids, True)
    b = [torch.zeros(n, dtype=t, device="cuda") for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    h["spnet"].export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
    es, _, ew, _, et = (x.cpu().numpy() for x in b)
    got = {(int(s), int(t)): float(w) for s, t, w in zip(es, et, ew)}

    base = {S.EXC: S.N_EXC, S.INH: S.N_INH, S.INP: S.N_IN, S.OUTP: S.N_OUT}
    exact = wrong = missing = 0
    for c, g in enumerate(genomes):
        gs_ = np.empty(g["weight"].size, np.int64)
        gt_ = np.empty_like(gs_)
        for p in (S.EXC, S.INH, S.INP):
            m = g["src_pool"] == p
            if m.any():
                gs_[m] = ids[p][c * base[p] + g["src_idx"][m]]
        for p in (S.EXC, S.INH, S.OUTP):
            m = g["tgt_pool"] == p
            if m.any():
                gt_[m] = ids[p][c * base[p] + g["tgt_idx"][m]]
        for i in range(g["weight"].size):
            k = (int(gs_[i]), int(gt_[i]))
            if k not in got:
                missing += 1
            elif abs(got[k] - float(g["weight"][i])) <= 1e-3:
                exact += 1
            else:
                wrong += 1
    print(f"K={K}: PLACEMENT exact={exact:,}/{exact + wrong + missing:,} "
          f"wrong={wrong} missing={missing}")
    del h
    torch.cuda.empty_cache()

# ---- 2. timing at the real K=128 / engine gs=128 scale
genomes = [S.seed_genome(np.random.default_rng(i), 30.0) for i in range(128)]
t0 = time.time()
h = S.build_pool(genomes, "cuda", seed=1, stdp_lr=0.01, w_max=30.0)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"K=128: build_pool via STOCK path {dt:.1f}s ({h['n_syn']:,} synapses)  "
      f"-> {'FAST (was 77s round with the host loop)' if dt < 30 else 'STILL SLOW'}")
