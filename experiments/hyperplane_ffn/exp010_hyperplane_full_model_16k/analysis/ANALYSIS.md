# exp010 learned-hyperplane geometry

Analysis of the learned `HyperplaneMultiHeadLUT` parameters in exp010's final
checkpoint (best val_bpb 1.1940, 324.7M params, anchor-pair init). Read-only,
CPU-only. Numbers are pooled over the 6 layers per site (emb_resid is a single
module). Full per-site stats in `geometry_stats.json`; bar charts in `geometry.png`.

Input dim d = 384 at every site. Random-Gaussian baseline for E|cos| of two rows
in R^384 is `sqrt(2/(pi*d)) = 0.0407`.

| site | K (nap) | rows | within-table mean\|cos\| | rand base | erank/K | PR (÷) | top-2 mass | ‖w‖ | \|b\| |
|------|---------|------|--------------------------|-----------|---------|--------|-----------|------|------|
| qk_lut       | 4 | 36864 | 0.0247 | 0.0407 | 1.00 | 2.5 | 0.901 | 1.479 | 0.0155 |
| v_lut        | 6 | 55296 | 0.0212 | 0.0407 | 0.99 | 2.4 | 0.919 | 1.480 | 0.0111 |
| out_proj     | 7 | 21504 | 0.0226 | 0.0407 | 0.99 | 2.4 | 0.907 | 1.465 | 0.0122 |
| residual_lut | 6 |  9216 | 0.0221 | 0.0407 | 0.99 | 2.3 | 0.928 | 1.441 | 0.0114 |
| emb_resid    | 6 |  1536 | 0.0149 | 0.0407 | 1.00 | 2.2 | 0.960 | 1.532 | 0.0136 |

Anchor init reference: each row is exactly 2-sparse (±1), so ‖w‖=√2≈1.414, b=0,
PR=2, top-2 mass=1.0.

## Headline findings (reconstruction-independent — computed from learned weights only)

1. **ORTHOGONALITY (the headline question): within a table the hyperplanes are
   near-orthogonal, and MORE orthogonal than random.** Mean |cos| between the K rows
   of a table is 0.015–0.025 across all sites — roughly **half** the random-Gaussian
   baseline (0.041) and close to 0. Mean *signed* cos ≈ 0. So the K sign-tests inside
   a table are close to mutually orthogonal, not aligned or anti-aligned.

2. **FULL RANK: the K rows of each table span a K-dim subspace (no collapse).**
   Effective rank (singular-value entropy) per table is ≈0.99–1.00 × K at every site.
   The hyperplanes within a table are not collapsing onto a shared direction.

3. **STILL ~2-SPARSE despite being free to densify.** Even though w receives dense
   gradients, the learned rows stayed concentrated: participation ratio 2.2–2.5 (init
   2.0) and **90–96% of ‖w‖² lives in the top-2 coordinates**. The learned front-end
   keeps doing essentially 2-coordinate comparisons — it did not spread into dense
   384-dim hyperplanes. Combined with (1), the near-orthogonality of ~2-sparse rows
   means the K bits of a table mostly test **disjoint coordinate pairs**.

4. **NORMS grew slightly, BIASES moved off zero but stayed tiny.** ‖w‖ ≈ 1.44–1.53
   (init √2≈1.414) — a modest increase. Biases started at exactly 0 and are now small
   but nonzero (mean\|b\| ≈ 0.011–0.016, mean b ≈ 0), i.e. the decision hyperplanes
   shifted marginally off the origin but remain nearly origin-centered.

## Drift from the specific anchor pair — UNCERTAIN (flagged, not asserted)

Reconstructing the anchor-pair init per module (deterministic in each module's seed;
verified the init is 2-sparse ±1, and that the assignment is seed-dependent) and
comparing learned→init: learned rows' top-2 coordinates match the reconstructed init
pair at only **chance level (~0.01)** and cos-to-init ≈ 0.

I cannot cleanly distinguish two explanations, and the run did **not** save an init
snapshot, so this stays uncertain:
- (a) the hyperplanes relocated off their initial coordinate pairs (staying ~2-sparse
  but on different coords), or
- (b) my seed-based reconstruction doesn't reproduce the exact per-row pair assignment
  the training run used.
A correct-seed vs deliberately-wrong-seed drift comparison were indistinguishable
(both ≈0), which is consistent with either reading. **Not reporting a drift/rotation
number as fact.** All findings in the section above are independent of this and stand.

## Notes
- Everything is pooled over layers; per-site the 6 layers were mutually consistent
  (no single layer drove a metric).
- Sanity: init reconstruction gives exactly-2-nonzero ±1 rows, ‖w‖=1.4142.
