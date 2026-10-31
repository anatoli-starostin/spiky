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

## Dominance on the ConfidenceLUT(n=1) cartridge (follow-up, 4000 steps, 1 seed each)
`dominance_conf.py` `DominanceConfFast` (option `dom_conf=True` of `DominanceStackLUT`): each table reads its single addressed
row (nop=8 deltas, hard detached address) and multiplies it by the library's n=1 confidence score
`s = (Σ_j m_j)·exp(γ Σ_j log σ(β m_j))`, `m_j=|u_j|` the table's 8 probe margins; `log β`, `log γ` are learnable scalars per layer
(init 2, 1, exactly `ConfidenceBase`). Borda aggregation within heads as in DominanceLUT (dense signed-incidence bmm of
`dominance_fast`), then `nn.LayerNorm(48)` with affine, then layer 2. Same anchors/pairs as before (tph 64, nap=nop=8,
`output_anchors='same'`, seeds as B). **Gradients:** no two-cell rational surrogate (that is the hard-Dominance one); as in
ConfidenceLUT n=1 the input gradient reaches `z` only through `s` (all 8 margins), tables get `s·g_row`, β/γ through `s`. The score is
computed in autograd with the library's own `ConfidenceLUT._score` (unbound; backward into margins, β, γ and `z[a]-z[b]` is
autograd's); the row read × score × Borda is one `autograd.Function` with hand-written backward (`g_row=gy@Mᵀ`, `dW=index_add(s·g_row)`,
`ds=<g_row,row>`). No table dropout / regulariser on the Dominance layer (as in B).
**Tests** (`test_dominance_conf.py`, 49 pass in total with `python -m pytest -q .`): value and gradients (tables, log β, log γ, input) vs an
independent float64 plain-PyTorch reference with explicit loops over heads/tables and pairs (3 geometries, CPU and CUDA);
fast vs the module's plain-torch path at the experiment geometry; the score equals `FusedConfidenceLUT` n=1's (outputs of an
all-ones table, same anchors) and `ConfidenceLUT._score` bit-exactly; zero-sum Borda output; stacks with `FusedManifestoHard` and `FusedConfidenceLUT` n=1 as layer 2.
**Init scale:** s ≈ 0.2 at init, so the unscaled Borda output had std 0.265 (per layer 0.265–0.268, on N(0,1) tokens through the
init compress projection); `train.py` (`dom_calibrate`) multiplies the delta tables by 1/std (×3.73–3.77, a private RNG; global
RNG stream unchanged) → pre-LN std 1.000 in every layer at init.

Runs (`configs/{E1_domconf_ln_fmh,E2_domconf_ln_conf1,F_conf_n1}_4k.json`, launcher `run_4k_conf.sh`, one sequential job; same data/seed/schedule;
W&B project `Spiky`, group `lutorch_ex_fused_dominance_stack`; figure `val_bpb_4k_conf.png`, `plot_curves_conf.py`; `results/`):
E1 = DominanceConf → LN(affine) → FusedManifestoHard rational (as B1 layer 2); E2 = DominanceConf → LN(affine) → FusedConfidence n=1
(layer 2 with β,γ init 2,1, dropout 0.2); F = single FusedConfidence n=1. (`train.py` fix: `conf_n1` used to pass `read_tau_*`, which n=1 rejects.)

| | val_bpb @4000 | ms/step | peak GB | params | table params |
|---|---:|---:|---:|---:|---:|
| A FusedManifestoHard rational | 1.36198 | 141.4 | 7.69 | 68.24M | 37.75M |
| B Dominance(hard) → MH | 1.40976 (rerun 1.40700) | 202.0 | 7.79 | 74.53M | 44.04M |
| B1 Dominance → LN → MH | 1.40608 | 222.4 | 7.79 | 74.53M | 44.04M |
| C FusedConfidence n=2 | 1.33472 | 152.7 | 7.69 | 68.24M | 37.75M |
| D MH(tph 11) → MH | 1.40335 | 163.4 | 7.79 | 74.73M | 44.24M |
| **E1 DominanceConf → LN → MH** | **1.40199** | 239.6 | 7.81 | 74.53M | 44.04M |
| **E2 DominanceConf → LN → Conf n=1** | **1.38342** | 242.1 | 7.81 | 74.53M | 44.04M |
| **F FusedConfidence n=1** | **1.33343** | 154.2 | 7.69 | 68.24M | 37.75M |

(ms/step: mean over steps >10; earlier rows are not rerun, timings vary between sessions by ~5%.) Noise at 4k ≈ 0.003 (B rerun).

Diagnostics at each eval (`hidden_std_pre_ln`, `score_mean` in the csv/W&B; means over tokens, heads, 6 layers): pre-LN std is 1.0 at init,
then 2.4 (E1) / 2.4 (E2) at step 250, peaks 4.8 / 5.6 (steps 750–1250) and ends 3.8 / 5.3; per layer at 4000 E1: 2.0, 4.4, 5.2, 4.6, 4.1, 2.8.
The score mean s: 0.46 at step 250, max 0.83 (E1, step 750), 0.50 (E1) / 0.59 (E2) at 4000. So the scale grows ×4–5 after init (the compress
projection widens the margins and s is superlinear in them); LN removes it and the init rescale was not what mattered.

**Read.**
* Confidence in the Dominance layer does **not** close the gap to A: E1 1.402 vs A 1.362 (+0.040), E2 1.383 (+0.021). It is nowhere near F/C
  (E1 +0.069, E2 +0.050 vs F; F is 0.001 better than C n=2 here, within noise, and 0.029 better than A).
* E1 vs D (depth control, two hard-ish layers, no dominance, 1.40335): E1 − D = −0.001, i.e. no difference (noise 0.003); E1 vs B1 −0.004 (≈noise).
  Putting the confidence score on the Dominance layer's first layer does not help when layer 2 is hard: the stack is still about as bad as any 2-layer hard stack (+0.04 vs A).
* E1 vs F: the 2-layer stack with a confidence first layer is 0.069 worse than the single confidence layer, at 1.55× the step time and 6.3M more params.
* E2 vs E1: −0.0186, 4–6× the noise, so layer 2 matters: a confidence layer 2 recovers about half of the depth penalty (E2 is still +0.021 vs A, +0.050 vs F).
  Layer 1 and 2 together are not additive-good: the best 2-layer stack is still worse than the single confidence cartridge. (No E2' with a hard layer 1 and confidence layer 2 was run, so how much of E2's gain is layer 2 alone is not separated.)
* Not established: 1 seed, 4k steps (ranking at 48k can differ), no 48k runs; Dominance-layer hyperparameters (delta init, lr, TV, no table dropout) untuned.

## E2 at 48000 steps (report-style schedule, 1 seed)
`configs/E2_domconf_ln_conf1_48k.json` = the 4k E2 config with only `n_steps` 48000, `eval_every` 500 (and `exp_name`); lr 3e-4, warmup 0.1,
cosine to 0.1·lr, 24576 tokens/step, seed 1. Launcher `run_48k_E2.sh` (checkpoint every 2000 steps, resumable; it did not need to resume),
W&B project `Spiky`, group `lutorch_ex_fused_dominance_stack`, run `dominance_stack_E2_domconf_ln_conf1_48k_main`. Figure `val_bpb_48k_E2.png`
(`plot_curves_48k.py`); csv/summary in `results/E2_domconf_ln_conf1_48k_*`. The report's per-step curves are not in the repo, so the report's
final 48k values are drawn as horizontal lines.

| | val_bpb @48000 |
|---|---:|
| dense FFN (report) | 1.08936 |
| FusedConfidenceLUT n=2 (report) | 1.08653 |
| FusedConfidenceLUT n=1 (report) | 1.09698 |
| FusedManifestoHardLUT rational (report) | 1.13242 |
| FusedManifestoHardLUT soft-sign (report) | 1.13780 |
| **E2 DominanceConf → LN → Conf n=1 (this run)** | **1.11984** |

E2 48k: 3.51 h, mean 240.8 ms/step (median 240, p99 281 over the logged steps; one logged outlier of 1159 ms), peak GPU memory 7.81 GB,
74.53M params (44.04M table params). val_bpb: 500: 2.286, 1000: 1.924, 2000: 1.706, 4000: 1.477, 8000: 1.284, 16000: 1.208, 24000: 1.174,
32000: 1.149, 40000: 1.130, 44000: 1.123, 48000: 1.1198 (monotone decreasing). E2 crosses the report's hard-LUT rational line at ≈ step 38.5k.
Diagnostics: pre-LN std 1.5 (500) → 7.5 (peak ≈16k) → 6.5 (48000); mean score s 0.32 → 0.78 (≈16k) → 0.67; at 48000 per layer pre-LN std
3.3, 6.4, 8.8, 7.4, 6.9, 5.8.

**Read.**
* vs the 4k runs (4k: A 1.36198, E2 1.38342, F 1.33343): at 4k E2 was +0.021 vs A (worse than the hard LUT); at 48k E2 is **−0.0126 vs
  FusedManifestoHard rational (1.13242)** and −0.018 vs soft-sign, so the 4k ranking (E2 worse than A) reverses with the long schedule.
* vs the confidence cartridge: E2 is +0.0229 vs ConfidenceLUT n=1 (1.09698), +0.0333 vs n=2 (1.08653) and +0.0305 vs dense (1.08936).
  At 4k the gap E2 − F was +0.050; it shrinks to +0.023 at 48k but the single confidence layer is still clearly better, at 1.57× the step time (240.8 vs ~154 ms)
  and 6.3M more params.
* Not established: 1 seed (report-quoted noise at 48k ≈ 0.001, much smaller than all gaps above); report numbers come from the report runs,
  not re-run here (different session/hardware timing is irrelevant to bpb but not excluded); no 48k run of E1/F in this setup.
