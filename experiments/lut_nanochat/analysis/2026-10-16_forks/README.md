# LUT-FFN d24 forks of v2 — 2026-10-16/17

Four single- or two-variable forks of `d24-lutffn-confn1-1xh100-s0-full-v2` (h16/tph64/d48, r=768, nap8,
read_top_n=1, fp8 backbone, fp32 projections, `--no-lr-rescale`, 5568 steps). All four were stopped early
(the machines were reclaimed); none reached the end of training or the standalone base_eval.

| run (wandb id) | change vs v2 | launcher | stopped at | GPU-h | s/step |
|---|---|---|---|---|---|
| `d24-lutffn-confn1-h32-tph32-1xh100-s0-7f4cc747` | h32/tph32 (r=1536), bf16 projections | `runs/lut_ffn_d24_h32_tph32_1xh100.sh` | 3778 | 19.6 | 17.20 |
| `d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0-04a80c7b` | the above + nap=9 (512 cells) | `runs/lut_ffn_d24_h32_tph32_nap9_1xh100.sh` | 2580 | 13.9 | 17.85 |
| `d24-lutffn-confn2-1xh100-s0-f1aad8a0` | read_top_n=2 | `runs/lut_ffn_d24_confn2_1xh100.sh` | 2133 | 11.0 | 17.18 |
| `d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0-036e09af` | LR rescale on (1.4142) + `--lut-ffn-proj-fp8-muon` | `runs/lut_ffn_d24_projmuonfp8_lrrs_1xh100.sh` | 2098 | 9.1 | 14.03 |

v2 itself: 14.544 s/step, final val bpb 0.7794. Dense baseline `d24-dense-1xh100-s0`: final val bpb 0.7193, CORE 0.2547.

## Val bpb at shared eval points (delta vs v2; negative = better)

| step | dense | v2 | confn2 | Δ | projmuon | Δ |
|---|---|---|---|---|---|---|
| 250 | 0.99236 | 1.08140 | 1.06030 | −0.02109 | 1.05037 | −0.03103 |
| 500 | 0.90220 | 0.96624 | 0.96248 | −0.00376 | 0.96062 | −0.00562 |
| 750 | 0.86512 | 0.92373 | 0.92149 | −0.00224 | 0.92179 | −0.00194 |
| 1000 | 0.84546 | 0.90038 | 0.89905 | −0.00132 | 0.90063 | +0.00026 |
| 1250 | 0.83205 | 0.88530 | 0.88441 | −0.00089 | 0.88578 | +0.00049 |
| 1500 | 0.82286 | 0.87487 | 0.87399 | −0.00088 | 0.87661 | +0.00174 |
| 1750 | 0.81599 | 0.86611 | 0.86521 | −0.00090 | 0.86858 | +0.00247 |
| 2000 | 0.80895 | 0.85859 | 0.85795 | −0.00065 | 0.86070 | +0.00211 |

- **confn2 (read_top_n=2):** early lead that converged to ≈ v2 (−0.0006 at 2000); no measurable lasting gain; +18% s/step.
- **projmuon-lrrs:** ahead until ~step 1000, then consistently behind (+0.002 bpb, +0.010 nats train loss); the two
  changes are confounded. Muon keeps the projections near full rank (decompress effective rank 702/768,
  σmax/σmin ≈ 8, vs AdamW's 269 and ≈ 1,800 at the same step) — mechanism confirmed, no val-bpb benefit by 2000. 4% faster.
- **nap=9 vs nap=8 (h32/tph32):** nap9 lower at every shared point, but the gap narrowed from −0.0037 (1500) to −0.0024 (2500).
- In-training CORE is a 500-example subsample (noise ~0.008); not conclusive for any of these.

## Notes / gotchas found

- `runs/lut_ffn_d24_1xh100.sh` (v2's launcher) does not pin the horizon; v2 got 5568 from `NUM_ITERATIONS` in the
  environment. Without it base_train auto-derives a shorter horizon. The newer launchers pin it.
- The in-training checkpoint pruner in `runs/baseline_d24_1xh100.sh` died on its first pass under `set -euo pipefail`
  (fixed in 4ffc6643); before that KEEP_CKPTS was only applied once, at the end of training.
- The LR schedule is warmup 40 / warmdown 65% / final 0.05: the decay starts at ~35% of training.
- The fused ConfidenceLUT kernel takes nap and read_top_n as runtime arguments (nap ≤ 16, n ∈ {1,2}); verified
  bitwise/1e-7 against the reference at nap=9 and at n=2 (forward and backward).

## Contents

- `gates/` — pre-launch probes: fused kernel vs reference at nap=9 and at read_top_n=2 (fwd + bwd), fp8 projection sanity.
- `optimizer/` — CPU-only optimizer-group coverage (confn2) and Muon routing of the projections (projmuon).
- `checkpoints/` — CPU-only readers: per-layer β/γ/τ values, projection-weight spectra.
- `curves/` — log parsers and the step-aligned comparisons above.
- `ops/` — launch / watch / stop / tag scripts and the worker sidecar pruner used for these runs.
