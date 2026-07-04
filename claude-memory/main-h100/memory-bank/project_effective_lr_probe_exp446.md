---
name: effective-lr-probe-exp446
description: exp446 measured the REAL Adam per-step lr (nominal × m̂/(√v̂+ε)) per layer/module in the exp428 LUT-LM. LUT realizes only ~22% of nominal 1e-3; lr variance is within-entry not between-row; row-starvation signature is weak (corroborates exp445).
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Effective (real) Adam lr probe — exp446 (2026-05-20)

Instrumentation fork of exp428 (config identical, final val 1.4978 ≈ exp428 1.4983, so no perturbation). Logged per (layer, module) every 200 steps: `nominal_lr` (base × cosine), `adam_factor = rms(m̂/(√v̂+ε))` ∈[0,1] (fraction of nominal lr Adam realizes), `eff_step_rms = nominal × factor` (real per-step displacement), plus a nested-ANOVA variance decomposition of |f| over the LUT weight tensor [n_tables, K=2^NAP, n_outputs]. Files: `nanochat_exps/exp446_effective_lr_probe/{effective_lr.csv, lut_lr_variance.csv, effective_lr_analysis.png}`; analysis script `/tmp/analyze_eff_lr.py`.

## Finding 1 — real lr ≈ 22% of nominal for ALL LUT modules
Converged `adam_factor`: qkv/v/out/residual all ≈ **0.22** (within 0.001 of each other) → real peak lr ≈ **2.2e-4** despite nominal `lut_lr=1e-3`. Adam damps the noisy per-row/STE gradients to ~⅕. Other groups: tok_emb **0.15** (most damped, sparse token rows, real ~4.6e-5); unembedder 0.55→**0.21** (consistent early, decays); temps **0.41–0.47** (highest); norms 0.30. Real-lr curves just trace the cosine schedule (factor ~flat post-warmup). **Takeaway: the LUT's nominal 1e-3 is mostly illusory — real motion ≈ 2.2e-4.**

## Finding 2 — lr variance is WITHIN-ENTRY, not between-row
Nested-ANOVA at step 8000 (avg over layers): within_entry **91–97.5%**, between_row 2.5–7.9%, between_table <0.5%. The lr spread lives across the n_outputs of a row (different output channels have different gradient SNR — generic), NOT across rows. The hot/cold-row "starvation" signature is small AND shrinks over training. (Earlier smoke-test read of "out_proj = between-row dominated" was a t=3 init transient — WRONG at steady state.)

## Finding 3 — depth matters only transiently
out_proj between_row fraction by layer: deep layers L4/L5 start uneven (0.30/0.40 at step 200) but all converge to ~0.02–0.03 by step 2000. At convergence the Adam factor is uniform across all 6 layers (~0.22) for every LUT module — no layer systematically starved.

## Strategic tie-in
Weak/decaying between-row lr variance ⇒ per-row coverage is actually decent at bs=16-effective (Adam is per-element scale-invariant: a cold row still takes a normalized step when hit; factor only decays for long-cold rows). This **corroborates [[soft-wgrad-neutral-exp445]]**: dense soft weight-grad gained nothing because per-row sparsity is NOT the bottleneck. Consistent with [[hard-forward-is-the-goal]] direction — the gap to soft-forward is functional, not an optimization/coverage problem.
