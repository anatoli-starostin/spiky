---
name: project_lut_convergence_bottleneck
description: "Measured root cause of LUT-LM's slower-than-vanilla convergence at fixed batch — per-parameter gradient sparsity (each LUT row updated by ~80 tokens vs all 8192 for a vanilla weight) → ~2.2x noisier body gradients."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Question (2026-05-22):** why does LUT-LM (exp475, 89.4M) converge to WORSE loss
than vanilla (exp476, 35.8M) at the same bs=16, despite MORE params? (1.4962 vs
1.4143.) Note: convergence speed, NOT wall-clock.

**Answer: per-parameter gradient sparsity (low gradient density).** Measured in
`nanochat_exps/profile_lut_vs_vanilla/grad_snr.py`.

**Mechanism (measured, gradient density = tokens contributing to each weight per
bs=16 microbatch of 8192 tokens):**
- vanilla `Linear` weight: gradient from ALL 8192 tokens (dense).
- LUT weight-row: gradient only from the ~80-100 tokens whose hard-argmax routing
  selects it → **~80x fewer effective samples per parameter**.
- Per-module median tokens/row: qkv/v/residual ~100 (p10~35, ~2% rows <16 tokens);
  **out_proj 79 (p10=9, 14% of rows get <16 tokens/step)** — worst module, matches
  the all-session out_proj-collapse finding.
- IMPORTANT: a "bs=16" microbatch is 16 seqs x 512 = 8192 TOKENS, so rows are nearly
  fully COVERED (qkv/v/residual coverage 1.0, out_proj ~0.9). The bottleneck is NOT
  uncovered rows — it's few SAMPLES per row (noisy estimate), not zero.

**Effect (gradient noise-to-signal E||g-mu||^2/||mu||^2 over independent bs=16
microbatches; higher=noisier=slower SGD):**
- at INIT: LUT body 0.65 vs vanilla body 0.29 → **LUT body 2.2x noisier**.
- at trained optimum: LUT body 14.5 vs vanilla body 8.8 → 1.65x noisier.
- both models' dense HEADS similar (0.70 vs 1.28 init) — confirms it's the BODY that
  differs (the dense head is also sparse-per-vocab-row, hence both noisy).

**Magnitude caveat:** naive "80x fewer samples → 9x noisier" OVERpredicts; measured is
~2.2x because tokens routing to the same row have CORRELATED gradients (routing groups
similar inputs), partially compensating for fewer samples.

**Why this is THE bottleneck & unifies the session:**
- structural, not capacity: LUT has 2.5x more params but learns slower — conditional
  routing spreads a fixed token budget across more, more-sparsely-updated params. Dense
  matmul (every param sees every token) is exactly what LUT trades away for matmul-free
  inference / conditional compute.
- explains why batch-scaling works (more batch = more tokens/row = higher per-row SNR,
  samples ∝ batch) and why every fixed-batch trick fails (can't add samples):
  [[project_bs48_what_improves]], [[project_prob_forward_dead_end]],
  [[project_soft_winner_dead_end]].
- to beat it token-efficiently you need either denser gradient per token (distillation:
  smooth teacher target hits all params) or a genuinely lower-variance estimator (none
  found exp353-488).

**CROSSOVER / "vanilla adapts faster late" (user obs, confirmed 2026-05-22):** comparing
exp486 (LUT bs=48) vs exp476 (vanilla bs=16) per-step trajectories: gap (van−LUT) PEAKS
early (+0.156 @ step 1000) then COLLAPSES (+0.081 @ 4k, +0.035 @ 8k). Late-stage slope
(same cosine schedule) 4k→8k: LUT −0.0159/1k vs vanilla −0.0266/1k → **vanilla descends
1.7× faster late**; extrapolated it crosses. exp486 still ENDS lower (1.3791<1.4143) but
only on banked early lead. Mechanism: TWO regimes —
  - early = gradient-QUANTITY-limited (coarse moves) → LUT bs=48's 3× tokens win.
  - late = gradient-PRECISION-limited (fine refinement near optimum) → per-param SNR
    matters, and even at bs=48 each LUT param gets ~315 samples/row vs vanilla bs=16's
    8192/weight (**26-31× fewer, MEASURED**) → LUT gradient too noisy to refine finely,
    plateaus; vanilla's dense clean gradient keeps descending.
**You cannot out-batch a density deficit:** batch buys QUANTITY (tokens/row), not DENSITY
(fixed at 1/K=1/64 by architecture). 3× batch chips 3× at a 64× gap, never changes density.
Fix late-stage adaptation by raising DENSITY: lower K (K=64→16 = 4× tokens/row, costs
expressivity/table — testable, the one lever that moves the measured number), or distillation
(dense teacher signal to all params via head). Soft weight-grad does NOT raise density (exp445
neutral: peaked softmax gives negligible grad to non-winner rows).

**Tooling:** `profile_lut_vs_vanilla/grad_snr.py` (gradient density + noise-to-signal,
LUT vs reconstructed vanilla MinimalGPT). GOTCHA: the nanochat dataloader REUSES one
buffer object per next() call — must `.clone()` collected batches or all "microbatches"
alias the same data (zero gradient variance, silent).
