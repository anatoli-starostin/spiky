"""UNIT CHECK: does the output-drive gene actually move first-spike timing?

Same net, same states, three drive scales. Lower drive should push first spikes LATER and, if
the mechanism is going to be useful at all, spread them over MORE distinct ticks (which is
what would break the ties). Prints the numbers rather than asserting a direction, so a null
result is visible instead of hidden.
"""
import numpy as np
import torch

import steady_state as S

X, Y, Xpool, Ypool, Xval, Yval = S.load(64, 0, 256)
enc = S.LatencyEncoder(Xpool)
Xb, Yb, _ = S.sample_batch(Xpool, Ypool, 64, 0, 0)
g = S.seed_genome(np.random.default_rng(0), 30.0)

print(f"{'drive':>6s} {'mean tick':>10s} {'std':>7s} {'min':>5s} {'max':>5s} "
      f"{'distinct/state':>15s} {'tie rate':>9s} {'silent':>8s}")
for d in (1.5, 1.0, 0.6, 0.3, 0.15):
    h = S.build_pool([g], "cuda", seed=1, stdp_lr=0.01, w_max=30.0,
                     drives=np.array([d]))
    first, _ = S.run_episode(h, Xb, enc, 200.0, train=False)
    torch.cuda.synchronize()
    t = first[:, 0, :]
    ties = S.tie_rate_per_member(first)[0]
    distinct = np.mean([len(np.unique(r)) for r in t])
    silent = float((t >= S.N_TICKS).mean())
    print(f"{d:6.2f} {t.mean():10.2f} {t.std():7.2f} {t.min():5.0f} {t.max():5.0f} "
          f"{distinct:15.2f} {ties:9.3f} {silent:8.2%}")
    del h
    torch.cuda.empty_cache()
