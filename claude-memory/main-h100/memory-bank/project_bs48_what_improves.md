---
name: project_bs48_what_improves
description: "exp486 (bs=48) beats exp475 (bs=16) by -0.117 bpb; deep analysis shows the gain is holistic (not modular), concentrated in the LAST 2-3 layers and on RARE tokens — not row coverage."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**exp486 (2026-05-22): bs=48 for 8K = new bs-scaled best at the 89.4M exp475 shape.**
Fork of exp475 (bs=16, 1.4962), ONLY change device_batch_size 16->48
(total_batch_size 24576, grad_accum=1, 3x tokens). **Final = 1.3791 @ 89.4M, 1.232 h.**
Δ=−0.117 vs exp475; also −0.035 BELOW the fair untied-vanilla target (exp476=1.4143)
— LUT-LM passes the vanilla quality bar (at 3x tokens). Crossed exp475's FINAL by step
2800 (35% of run). Checkpoint at `nanochat_exps/exp486_bs48_8k/checkpoint.pt`.

**Deep comparison vs exp475** (harness in `nanochat_exps/analysis_exp475_vs_exp486/`:
model_def.py, analyze.py, transplant.py, probe.py — all reusable; shared seed ⇒
identical anchors ⇒ row (table,r) = same bit-pattern in both, so weight comparison valid):

1. **NOT row coverage.** On real val data both models cover ~all LUT rows; `revived_rows=0`
   for every module (rows dead@475 that become alive@486). The "dead rows grow with depth"
   seen in a random-token dry-run was an ARTIFACT of random input — rejected on real data.

2. **NOT modular / localizable.** Transplant ablation (graft one exp486 module-group into
   exp475, measure bpb): EVERY partial graft is catastrophic (module:out_proj −1.28,
   qkv −0.67, unembedder −1.11, layer:0 −0.42; only ALL recovers +0.14). The two solutions
   are holistically COADAPTED in different weight basins (rel_w_change≈1.4, rownorm_corr≈0.05).
   No single module to target post-hoc. Rules out "just fix out_proj/last-layer" shortcuts.

3. **Depth: gain is in the LAST 2-3 layers.** Logit-lens (ln_final+unembedder on cumulative
   residual after each layer) bpb-vs-depth delta(486−475): L1 +0.075, L2 +0.075, L3 +0.048,
   L4 −0.021, L5 −0.086, L6 −0.136. Early layers are slightly WORSE in exp486; the entire
   advantage is built in the final layers. bs=48 learns deeper composition rather than making
   shallow features greedily logit-readable.

4. **Tokens: gain is on RARE tokens, monotonically.** Per-frequency bits/token delta:
   rarest bucket −1.16, ... commonest −0.16 (monotone with rarity).

5. **Position: ~uniform** (~−0.63 bits beyond pos 16; biggest at pos 1-4 = −1.1). NOT an
   in-context-learning effect — base modeling.

**Unifying interpretation:** bs=48's benefit concentrates on the HARD/RARE/DEEP paths that
are gradient-UNDERSAMPLED at bs=16. Rare tokens appear ~3x less per step → their LUT paths
get ~3x less gradient; late layers sit at the end of the longest backprop path → noisiest
small-batch gradients. Both starve at bs=16. Consistent with exp367 (bs gain = pure gradient
quality) and out_proj-L5 being the most collapsed module at bs=16 (rownorm_cv 0.56→0.16).

**Token-efficiency levers:**
- **Inverse-frequency loss weighting — TESTED, FAILED (exp487, 2026-05-22).** Fork of exp475
  (bs=16), train loss weighted by (smoothed_freq)^−0.5, capped 5×, freq-weighted mean=1, eval
  bpb standard. **+0.36 bpb WORSE than exp475, stable, no crossover** (killed step 600).
  Conceptual reason it can't work: bs=48's gain is ABSOLUTE (3× samples of EVERY token — common
  stay trained AND rare improve), but reweighting is ZERO-SUM within a fixed batch (give rare
  more ⇒ take from common). Standard bpb is common-token-dominated, so the reallocation directly
  hurts the metric. **Confirms bs-scaling's benefit is more absolute gradient signal, not better
  allocation** — same root cause as every other fixed-batch reallocation dead end this session
  (windowed grad, hard mining, prob/soft_winner forward). Code: `freq_weight_alpha` in
  exp487/train.py + precompute_token_freq.py (off by default, alpha=0).
- **Higher late-layer LR — TESTED, FAILED (exp488, 2026-05-22).** Fork of exp475 (bs=16),
  last 2 layers' (L4,L5) LUT params at lut_lr×3 (6e-4), early layers at 2e-4 (split into two
  LION groups). **Stable +0.01 to +0.014 bpb WORSE than exp475** across all evals 200–1600,
  no crossover (killed step 1600). NOT zero-sum (doesn't steal gradient), so it sidesteps the
  exp487 trap — but still fails because the late-layer deficit at bs=16 is gradient NOISE, not
  under-movement. A bigger step on a noisier gradient just amplifies the noise. Confirms (again)
  bs-scaling's gain is gradient QUALITY/variance-reduction, unreachable by step-size or
  allocation tricks at fixed batch. Code: `late_lr_layers`/`late_lr_mult` in exp488/train.py
  (off by default, mult=1.0).

**OVERALL CONCLUSION (exp487+exp488):** the bs=48 gain is irreducible gradient-variance
reduction from MORE SAMPLES. Neither reallocating gradient between tokens (zero-sum, exp487)
nor scaling late-layer step size (non-zero-sum, exp488) recovers it — both fail for the same
underlying reason (the limit is noise, not allocation or magnitude). To get bs=48 quality at
fewer tokens you need a genuinely lower-variance gradient ESTIMATOR at fixed batch (none found
across the whole exp353–488 sweep), or just pay the tokens. grad_accum reproduces it exactly
[[project_grad_accum_reproduces_big_batch]] but at the same token cost.
