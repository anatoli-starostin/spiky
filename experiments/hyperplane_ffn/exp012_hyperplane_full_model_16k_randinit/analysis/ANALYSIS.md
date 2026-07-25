# exp012 (random-init) learned-hyperplane geometry

Identical analysis to exp010, on exp012's final checkpoint (best val_bpb 1.1953,
random hyperplane init, scale 0.05). CPU-only, read-only. d=384; random-Gaussian
|cos| baseline 0.0407. Pooled over 6 layers per site.

| site | K | mean\|cos\| | rand | erank/K | PR | top-2 | ‖w‖ | \|b\| |
|---|---|---|---|---|---|---|---|---|
| qk_lut | 4 | 0.0428 | 0.0407 | 0.99 | 129.2 | 0.048 | 1.076 | 0.0161 |
| v_lut | 6 | 0.0400 | 0.0407 | 0.99 | 129.2 | 0.048 | 1.073 | 0.0113 |
| out_proj | 7 | 0.0400 | 0.0407 | 0.99 | 128.2 | 0.049 | 1.063 | 0.0132 |
| residual_lut | 6 | 0.0376 | 0.0407 | 0.99 | 129.5 | 0.048 | 1.038 | 0.0145 |
| emb_resid_lut | 6 | 0.0352 | 0.0407 | 0.99 | 129.7 | 0.048 | 1.142 | 0.0136 |

## Headline: random-init converges to a DENSE, random-like hyperplane geometry
(the OPPOSITE of exp010's sparse geometry — yet essentially the same val_bpb).

- **NOT sparse.** Participation ratio ~128-130 (a dense iid-Gaussian row in R^384 has
  PR≈d/3≈128), and only ~4.8% of ‖w‖² sits in the top-2 coordinates. The hyperplanes
  stayed spread across ~1/3 of all input dims — they did NOT sparsify toward the
  2-coordinate comparators exp010 uses.
- **Orthogonality at the random baseline.** Within-table mean|cos| ≈ 0.035-0.043 ≈ the
  0.041 random-Gaussian baseline (exp010 was ~0.02, well below it). So exp012's rows
  are only as decorrelated as random dense vectors, not extra-orthogonal.
- **Full rank** (erank/K≈0.99) — same as exp010 (trivial for dense rows).
- **Norms** ~1.04-1.14 (random init ‖w‖=0.05·√384≈0.98, grew modestly). **Biases**
  small (|b|≈0.011-0.016), same order as exp010.
- Drift-vs-init ≈0 (same reconstruction caveat as exp010; in R^384 even small rotations
  drop cos to ~0, so this is uninformative — not asserting a rotation).

## Read
exp012 kept the DENSE random character of its init; exp010 kept the SPARSE anchor
character of its init. The two runs land at nearly the same val_bpb (1.1953 vs 1.1940)
via completely different hyperplane geometries. See COMPARISON.md.
