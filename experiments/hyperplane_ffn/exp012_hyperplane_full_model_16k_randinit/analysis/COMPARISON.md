# exp010 (anchor init) vs exp012 (random init) — hyperplane geometry

Both are the identical ~325M full-model HyperplaneMultiHeadLUT (exp752 geometry/recipe,
16000 steps); the ONLY difference is hyperplane init. Final results are essentially
tied: exp010 best 1.1940, exp012 best 1.1953 (Δ+0.0013); both beat exp752 (fixed
anchors, 1.2162) and exp001 (dense MLP, 1.20144).

Despite the near-identical loss, the LEARNED hyperplane geometries are opposite:

| metric (pooled) | exp010 anchor-init | exp012 random-init | random baseline |
|---|---|---|---|
| participation ratio (density) | ~2.2-2.5 (SPARSE, ~2 coords) | ~128-130 (DENSE, ~d/3) | ~128 |
| top-2 coord mass | 0.90-0.96 | ~0.048 | ~0.016 |
| within-table mean\|cos\| | 0.015-0.025 (below random) | 0.035-0.043 (= random) | 0.041 |
| effective rank / K | ~1.0 | ~1.0 | ~1.0 |
| ‖w‖ | 1.44-1.53 | 1.04-1.14 | init 0.98 |
| \|b\| | 0.011-0.016 | 0.011-0.016 | 0 |

**Conclusion.** Each run stays in the geometric basin of its init: anchor init keeps
~2-sparse, extra-orthogonal "coordinate-pair comparator" hyperplanes; random init keeps
dense, random-orthogonality hyperplanes. They reach the same performance by different
structures. So (a) exp010's learned-hyperplane win is robust to init (holds from random),
and (b) the sparse coordinate-pair structure is NOT necessary for the performance — a
dense hyperplane front-end does just as well. The anchor prior shapes the solution's
geometry (and speeds early convergence) but is not required for the final quality.
