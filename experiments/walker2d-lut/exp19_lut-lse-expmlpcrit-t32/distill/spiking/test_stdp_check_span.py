"""CPU-only: confirm --one-meta-per-source really leaves no source spanning two metas,
and report the per-(meta,source) sublist sizes that the native grow will see."""
import numpy as np

NE, NI = 64, 16

for n_plastic in (1, 2, 4, 20):
    for ompv in (True, False):
        rng = np.random.default_rng(0)
        E, I = list(range(1, NE + 1)), list(range(NE + 1, NE + NI + 1))
        keep = {}
        for _ in range(NE * 16):
            s, t = E[rng.integers(NE)], E[rng.integers(NE)]
            if s != t:
                keep[(s, t)] = (s % n_plastic if ompv else int(rng.integers(0, n_plastic)))
        for _ in range(NI * 8):
            s, t = I[rng.integers(NI)], E[rng.integers(NE)]
            keep[(s, t)] = n_plastic
        metas_of = {}
        sub = {}
        for (s, t), m in keep.items():
            metas_of.setdefault(s, set()).add(m)
            sub[(m, s)] = sub.get((m, s), 0) + 1
        span = sum(1 for s, ms in metas_of.items() if len(ms) > 1)
        sizes = np.array(list(sub.values()))
        print(f"  n_plastic {n_plastic:2d}  one_meta_per_source={str(ompv):5s} -> "
              f"sources spanning >1 meta: {span:3d}/{len(metas_of)}   "
              f"(meta,source) sublists: {len(sub):4d}  "
              f"sizes min {sizes.min()} max {sizes.max()}")
