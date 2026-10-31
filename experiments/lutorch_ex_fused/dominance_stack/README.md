# dominance_stack — does a HARD DominanceLUT in front of the hard LUT recover the ConfidenceLUT gap?

Question: replace each FFN's `FusedManifestoHardLUT(rational)` (report: 1.13242 @48k, dense 1.08936, conf n=2 1.08653)
by a per-head stack **DominanceLUT (hard read) → FusedManifestoHardLUT (rational)**. Everything else is the
`01_fmh_rational` recipe (`train.py` here is a copy with only the changes listed in its docstring).

## What is here
| file | what |
|---|---|
| `dominance_lut_ref.py`, `test_dominance_lut_ref.py` | verbatim copy of DeepSpike-AI/deepspike-ml `research/lut_capacity_theory/dominance_lut/dominance_lut.py` + its tests (only the import in one test renamed) |
| `dominance_fast.py` | `DominanceLUTFast`: Triton row-gather + dense signed-incidence Borda (bmm) + hand-written backward (tables, input surrogate gradient as in the reference). Same state_dict. Reference needed 9.5 ms fwd+bwd per call at G=8, tph=64, 6144 rows, 60% in torch gathers. |
| `test_dominance_fast.py` | fast vs reference: value, table grad, input grad, bias grad, several geometries incl. the experiment's; fallback path |
| `stack.py`, `test_stack.py` | `DominanceStackLUT` (layer 1 `DominanceLUTFast`, layer 2 any ManifestoLUT; `MultiHeadLUT` interface, plugs into `ProjectionMHL`), generic `LUTStack` (control). Test: stack == reference Dominance → FusedManifestoHard layer by layer, value and all grads. |
| `train.py`, `configs/`, `run_4k.sh`, `env.sh` | trainer, configs, the launcher used, environment |
| `results/`, `val_bpb_4k.png`, `plot_curves.py` | 4k run metrics/summaries, figure |

Tests: `source env.sh; python -m pytest -q .` → 39 passed (CUDA).

## Stack definition (B)
Per head (G=8, own table slices; both layers batch over G, no Python loop): layer 1 = DominanceLUT hard (`vote='delta'`),
d 48→48, tph 64, nap=nop=8, `output_anchors='same'`, init std `sqrt(d_out/(2·tph·nop))` (output std ≈1), input gradient
as in `dominance_lut.py` (two-cell rational surrogate); layer 2 = FusedManifestoHardLUT rational, tph 64, nap 8, same
faithful init (Uniform ±1e-3), cell-TV λ=10 and table dropout 0.2 as in the baseline. Anchor seeds: layer 2 `1000+i`
(identical to the baseline), Dominance `2000+i`, i = transformer layer. **No regulariser on the Dominance layer**
(not tested: TV on delta rows). Dominance outputs are zero-sum over the 48 outputs (property of the layer).
Cell TV is the mean over the model's Manifesto layers, i.e. layer 2 only.

## 4000-step results (1 seed, schedule compressed: warmup 10%, cosine to 0.1·lr; eval every 250)
| | val_bpb @4000 | ms/step (mean, steps>10) | peak mem GB | params | LUT table params |
|---|---:|---:|---:|---:|---:|
| A FusedManifestoHard rational | 1.36198 | 141.4 | 7.69 | 68.24M | 37.75M |
| **B Dominance(hard) + ManifestoHard** | **1.40976** | 202.0 | 7.79 | 74.53M | 44.04M |
| C FusedConfidence n=2 | **1.33472** | 152.7 | 7.69 | 68.24M | 37.75M |

B − A = **+0.0478 (worse)**; C − A = −0.0273. Curves (`val_bpb_4k.png`, `results/*_metrics.csv`): B is above A at every
eval from step 250 on (gap 0.03 at 250, ~0.04–0.05 from 1000 on, not closing), C is below A from step 750.
Smoke (40 steps, same measurement) gave 141/203/152 ms/step, so timings are stable. Stack costs 1.43× A per step, +0.1 GB.
W&B: project `Spiky` (team entity anatoli-starostin-relocation, same as the old runs), group `lutorch_ex_fused_dominance_stack`,
runs `dominance_stack_{A_fmh_rational,B_dom_fmh,C_conf_n2}_4k_*`.

**Read:** at 4k steps, with this configuration, a hard Dominance layer in front does NOT help; it hurts by 0.048 bpb
and costs 43% more time. It is far from approaching ConfidenceLUT.

## What is not established
* 1 seed; 4k steps only. Early-training ranking need not match 48k (in the report, conf vs hard gap is already visible by
  step 7.5k, but a stack could differ in the late phase). Seed noise at 48k was ~0.001, at 4k probably larger but surely < 0.048.
* Why it hurts: not diagnosed. Candidates: zero-sum Dominance output / low-rank rows feed a layer that only uses the sign of
  margins; the Dominance layer's outputs are bounded by 1-bit-addressed tables of std≈1 init and may be noisy for the 2nd
  layer's addresses; the stack is deeper without any regulariser on layer 1; lr / init not tuned for layer 1.
  Not tried: Dominance TV, `normalize='count'`, `output_anchors='random'`, soft/sign votes, different init std, residual (x + dom(x)).
* Depth vs dominance (control D) not run.
* Evaluation protocol is the repo's fixed_eval (same as the old runs); absolute 4k numbers are not comparable to 48k.

## Proposed 48k runs (NOT launched; need approval)
Configs: `configs/{A_fmh_rational,B_dom_fmh,D_fmh_fmh}_48k.json` (D: `fmh_fmh` = FusedManifestoHard(rational, tph=11, init
σ=1/√tph, no dropout) → FusedManifestoHard(rational, tph=64, baseline settings); layer-1 tables 1.08M ≈ Dominance layer's 1.05M,
total table params 44.24M vs B 44.04M). Estimates from the measured ms/step plus ~5.7 s per eval × 96 evals:
| run | ms/step | est. hours (1 H100) |
|---|---:|---:|
| A baseline (report has 1.13242 for the same recipe; rerun only if exactly the same code path is wanted) | 141 | ~2.0 |
| B stack | 202 | ~2.85 |
| D control | 152 | ~2.2 |
Total ≈ 7.1 GPU-h for all three (≈5.1 if A is skipped and the old 1.13242 is used; note the old runs used the same recipe on
the same corpus). Given the 4k result (B is 0.048 worse than A), my recommendation is to run D and B at 4k first (≈0.5 GPU-h)
or skip the 48k stack unless a variant (see "not tried") closes the gap at 4k. Launch: `CONFIG=configs/B_dom_fmh_48k.json OUT_DIR=runs/B_48k python train.py` after `source env.sh`.
