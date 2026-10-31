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
* Depth vs dominance: control D run in the LN follow-up below.
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

## LayerNorm after the Dominance layer (follow-up, 4000 steps, 1 seed each)
Option `post_norm` of `DominanceStackLUT` (`"none"` | `"ln"` | `"ln_noaffine"`): `nn.LayerNorm(48)` over the last dim of the
Dominance output `[B, G, 48]`, i.e. each head and token separately (one affine of 48 params per layer, shared over heads,
init w=1 b=0, no weight decay), before the second (ManifestoHard) layer. The Dominance output is exactly zero-sum per head,
so the mean is already 0 and LN is effectively a per-token rescale to unit variance. `train.py` additionally logs
`hidden_std_pre_ln` (std over the 48 coords, mean over tokens, heads, and the 6 layers, accumulated over each eval) to
csv/W&B. Configs `B1_dom_ln_fmh`, `B2_dom_lnna_fmh`, `D_fmh_fmh`, `B_dom_fmh_hstd` (= B config, only logging added);
launcher `run_4k_ln.sh` (sequential); figure `val_bpb_4k_ln.png` (`plot_curves_ln.py`), data `results/`.
W&B: same project/entity/group, runs `dominance_stack_{B1_dom_ln_fmh,B2_dom_lnna_fmh,D_fmh_fmh,B_dom_fmh_hstd}_4k_*`.

| | val_bpb @4000 | ms/step | params | table params |
|---|---:|---:|---:|---:|
| A FusedManifestoHard | 1.36198 | 141.4 | 68.24M | 37.75M |
| B Dominance → MH | 1.40976 | 202.0 | 74.53M | 44.04M |
| B (rerun, std logging only) | 1.40700 | 211.3 | 74.53M | 44.04M |
| B1 Dominance → LN(affine) → MH | 1.40608 | 222.4 | 74.53M | 44.04M |
| B2 Dominance → LN(no affine) → MH | 1.40908 | 217.7 | 74.53M | 44.04M |
| C FusedConfidence n=2 | 1.33472 | 152.7 | 68.24M | 37.75M |
| D MH(tph 11) → MH(tph 64), no LN | 1.40335 | 163.4 | 74.73M | 44.24M |

A, B, C are the earlier logged runs (not rerun). The code changed since (LN option, std logging in `train.py`/`stack.py`),
but the B path is numerically unchanged (post_norm none adds no RNG draws/ops); still, GPU kernels are not bitwise
deterministic: the B rerun gave 1.40700 vs 1.40976 (Δ0.0028, and different ms/step: 211 vs 202, timing varies between
sessions), so **run-to-run noise at 4k is ≈0.003** and differences below that are not resolved. B1 and B2 ms/step are not
exactly comparable to the earlier timings either (LN adds ~10 ms/step by itself: 222/218 vs 211 for the same-session B rerun).

**Hidden scale:** the Dominance output std before LN is 1.01 at step 250 and drifts up to 1.04–1.06 at 4000 (B 1.044, B1 1.062,
B2 1.063; per layer 1.0–1.1). The init target (≈1) is kept, so the scale of the Dominance output was not a problem and LN has
almost nothing to rescale.

**Read:** LN does not close the gap. B1 − B = −0.0037 vs the first B (−0.0009 vs the rerun),
B2 ≈ B; all within the ≈0.003 noise. Gap to A stays ≈ +0.044–0.048; gap to C ≈ +0.072–0.075. Control D (two
hard LUT layers, no dominance, no LN) gets 1.40335, the same as B/B1/B2 (within noise, even slightly better) and 0.041 worse than A, at
0.8× B's per-step cost: **the loss is explained by stacking a second LUT layer (depth) in this recipe, not by the Dominance
operator**; Dominance is neither better nor worse than a plain hard LUT as layer 1 here. Not established: 1 seed, 4k steps only,
no 48k runs were made.
