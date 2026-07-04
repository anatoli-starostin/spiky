---
name: soft-wgrad-neutral-exp445
description: "exp445 decomposes the exp444 soft-forward win: hard forward + soft weight-grad = exactly exp428 (hard/hard). The entire gain is the soft FORWARD, not dense weight-gradient coverage."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Soft weight-grad is exactly neutral; exp444's win is the soft FORWARD (2026-05-20, exp445)

exp444 (SoftMultiHeadLUT hard=False, full soft mixture forward) beat exp428 (TinyMHLut soft, hard argmax forward) by −0.0162 bpb @ 8K bs=16. exp445 decomposes which ingredient causes it.

## Setup
exp445 = fork of exp428 with `enable_soft_weight_grad(True)` (the `_GLOBAL_SOFT_WEIGHT_GRAD` toggle from exp360, path `_soft_lut_bwd_body_soft_w` in `tiny_multi_head_lut.py`). This keeps the HARD argmax forward (`W[chosen]`, 1 lookup/table) but attributes the WEIGHT gradient across all 2^NAP rows via `sel_soft` (`grad_w[k] += sel_soft[k]·grad_pt`) instead of `index_add` at the chosen row only. x-gradient unchanged (soft chain). Everything else identical to exp428 (NAP=6 everywhere, E=64/D=384/6L, qkv/v/out/residual tph=64/256/1024/128, bs=16 direct, 8000 steps, lut_lr=1e-3).

## Result (all 8K, 8192-token effective batch — fair per-step comparison)
| run | forward | weight grad | final bpb |
|---|---|---|---|
| exp428 | hard | hard (chosen only) | 1.498293 |
| **exp445** | hard | **soft (all 2^NAP rows)** | **1.498315** |
| exp444 | **soft** | soft (all rows) | 1.482111 |

- exp445 vs exp428: **Δ = +0.000022** — identical to 4 dp, pure eval noise.
- exp444 vs exp428: Δ = −0.016182 — the entire win.
- Trajectory: exp445 tracked exp428 within ±0.002 (eval noise) at EVERY eval 200→8000; never tracked exp444. exp444's late crossover (~step 3400, then widening to −0.016 during LR decay) did NOT appear in exp445.

## Conclusion
**Dense weight-gradient coverage has zero effect.** Spreading gradient across all rows (≈ Hamming-confidence-weighted neighbor sharing) does NOT help, even though it visibly attacks per-row sparsity. So bs=16 "row starvation" is NOT the bottleneck the soft forward fixes. The soft forward's gain is **representational/functional** (the blended output `Σ_k sel_soft[k]·W[k]` is simply a better function), NOT an optimization effect. Confirms exp360 at 89M base (there: tiny regression; here: exactly neutral).

**Practical:** exp444's −0.016 requires the soft mixture AT INFERENCE — cannot be recovered by a training-only weight-grad trick. Pays ~2^NAP lookups + softmax per table. exp445 ran 0.44h (cheap hard forward, native TinyMHLut kernels) vs exp444's 2.1h but bought nothing.

## Code clarification discovered during this run
The three soft backward bodies span TWO orthogonal axes (do not confuse):
- `_soft_lut_bwd_body` (backward_mode='soft'): weight grad HARD (chosen), x-grad over all 2^NAP.
- `_soft_lut_bwd_body_soft_w` (`_GLOBAL_SOFT_WEIGHT_GRAD=True`): weight grad SOFT (all rows), x-grad over all rows.
- `_soft_lut_bwd_body_topk` (backward_mode='soft_topk', topk_n_alt=n_alternatives): weight grad HARD (chosen, index_add lines 1214-1219), x-grad restricted to chosen + top-K lowest-|d| Hamming-1 neighbors.

So `soft_topk` truncates the X-gradient support, NOT the weight gradient. A "top-K soft WEIGHT grad" variant does NOT exist (would need masking sel_soft to top-K before the weight einsum at line 1118) — but given soft-wgrad is exactly neutral, that follow-up is very unlikely to help. The valuable next axis is the FORWARD: a top-K soft forward (blend chosen + top-K neighbors) to capture exp444's gain at K+1 lookups instead of 2^NAP. See [[soft-forward-beats-hard-exp444]].
