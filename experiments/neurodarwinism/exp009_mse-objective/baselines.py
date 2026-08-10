"""Baselines for exp009 and the cross-metric evaluation of exp008's tau-trained gated nets.

Without a baseline an MSE number is unreadable: the question is not "is it small" but "is it
smaller than predicting the same offset for every state".

The val set is drawn with default_rng(seed + 1) (data.py:40), so EACH RUN HAS ITS OWN
held-out sample and every number here has to be computed at that run's seed. The training
pool is seed-independent, so the target statistics are shared.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/neurodarwinism/src")
import steady_state as S                                   # noqa: E402
from data import load                                      # noqa: E402

E8 = "/home/astarostin/projects/spiky/experiments/neurodarwinism/exp008_output-delay-gate"
SEEDS = (0, 1, 2)
S.D_MAX, S.N_DELAY_METAS = 20, 20
S.OUT_GATE, S.OUT_D_MIN, S.OUT_D_MAX, S.N_OUT_DELAY_METAS = True, 64, 80, 17

print("=== BASELINE held-out MSE, per seed (offsets 0..31) ===")
print(f"{'seed':>4s} {'pool-mean const':>16s} {'best const':>11s} {'uniform random':>15s}")
base = {}
for s in SEEDS:
    _X, _Y, Xpool, Ypool, Xval, Yval = load(64, s, 2000)
    S.fit_target_stats(Ypool, 2.5, 32)
    tv = S.target_offsets(Yval)
    tp = S.target_offsets(Ypool)
    c = tp.mean()
    r = np.random.default_rng(0).integers(0, 32, tv.shape)
    base[s] = float(np.mean((tv - c) ** 2))
    print(f"{s:4d} {np.mean((tv - c) ** 2):16.3f} {np.mean((tv - tv.mean()) ** 2):11.3f} "
          f"{np.mean((tv - r) ** 2):15.3f}")
print(f"  (pool-mean constant is the honest baseline: it uses no held-out information)")

print("\n=== exp008 GATED nets, tau-trained, scored on BOTH metrics at their own seed ===")
rows = []
for s in SEEDS:
    ck = os.path.join(E8, f"gated_seed{s}", "ck.npz")
    if not os.path.exists(ck):
        print(f"  seed {s}: no checkpoint")
        continue
    _X, _Y, Xpool, Ypool, Xval, Yval = load(64, s, 2000)
    enc = S.LatencyEncoder(Xpool)
    S.fit_target_stats(Ypool, 2.5, 32)
    genomes, ewma, *_ = S.load_ckpt(ck)
    b = int(np.nanargmax(ewma))
    h = S.build_eval_pool(genomes[b], "cuda", 0.01, 30.0)
    ft, _, _, _ = S.score(h, Xval, Yval, enc, 200.0, 0.0, readout_window=32,
                          coverage_penalty=0.0, objective="tau")
    fm, _, _, _ = S.score(h, Xval, Yval, enc, 200.0, 0.0, readout_window=32,
                          coverage_penalty=0.0, objective="mse")
    rows.append((s, float(ft[0]), -float(fm[0]), base[s]))
    print(f"  seed {s}: member {b:2d}  corrected tau {float(ft[0]):+.4f}  MSE {-float(fm[0]):8.3f}"
          f"   (constant baseline {base[s]:.3f})")
    del h
    torch.cuda.empty_cache()
if rows:
    t = np.array([r[1] for r in rows])
    m = np.array([r[2] for r in rows])
    bl = np.array([r[3] for r in rows])
    print(f"  mean: tau {t.mean():+.4f} +/- {t.std(ddof=1):.4f}   "
          f"MSE {m.mean():.3f} +/- {m.std(ddof=1):.3f}   "
          f"baseline {bl.mean():.3f}")
    print(f"  -> tau-trained nets are {'BELOW' if m.mean() < bl.mean() else 'ABOVE'} the "
          f"constant baseline on MSE by {abs(m.mean() - bl.mean()):.3f}")
