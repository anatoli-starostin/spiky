---
name: soft-wgrad-and-contrast-dead-ends
description: "Two LUT-specific algorithmic ideas (soft weight-gradient backward via sel_soft, inter-table cosine-similarity contrastive loss) both failed to beat baseline at tiny LUT-LM scale."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Soft weight-gradient backward and inter-table contrastive: both dead ends (2026-05-15, exp360 + exp361)

After optimizer-side levers maxed out at ±0.01 bpb, we tried two algorithmic changes specific to the LUT structure. Both failed.

## exp360: Soft weight-gradient backward (killed at step 5400)

Replaced TinyMHLut's hard `index_add` weight gradient (`grad_w[chosen_row, :] += grad_y`) with sel_soft-weighted attribution across all K rows: `grad_w[row, :] += sel_soft[row] * grad_y`. Implementation: added `_GLOBAL_SOFT_WEIGHT_GRAD` toggle + `_soft_lut_bwd_body_soft_w` function in `src/spiky/lutorch/tiny_multi_head_lut.py`; opt-in via train.py `enable_soft_weight_grad(True)`.

| step | exp360 (soft wgrad) | exp353 (hard wgrad, same LR) | exp352 baseline | Δ vs exp353 |
|---|---|---|---|---|
| 200  | 2.3635 | 2.2945 | 2.3181 | +0.0690 |
| 800  | 1.9814 | 1.9174 | 1.9293 | +0.0640 |
| 2000 | 1.8065 | 1.7777 | 1.7901 | +0.0288 |
| 4000 | 1.7072 | (n/a)  | 1.6988 | +0.0084 (vs baseline) |
| 5400 | 1.6707 | (n/a)  | 1.6613 | +0.0094 |

**Pattern**: severely worse early, narrowing over time but never catches up; stable ~+0.007–0.010 worse than exp352 baseline by step 4000+. Soft weight-gradient was a regression.

**Why**: even though sel_soft sums to 1 so the total per-token L1 mass of weight gradient is conserved, the chosen row's effective LR is reduced (sel_soft[chosen]≈0.99 instead of 1.0 in steady state, less near init). The training "loses" magnitude on the genuinely useful row and gives small bias-noise to neighboring rows that won't be used at inference. The Adam state for the chosen row gets noisier (mixture of correct + small soft tail), and the "cold row" Adam updates from soft-tail are mostly noise, not signal.

## exp361: Inter-table cosine contrastive (killed at step 2800)

For each LUT module, computed pairwise cosine similarity between flattened table weights and penalized squared off-diagonal entries: `λ·mean(off_diag(cos_sim)²)`, λ=0.01.

Tables started near-orthogonal (init std=0.001 makes cosine similarity ~ 1/√(K·n_outputs) ≈ 0.01) so the contrast value stayed at ~0.005 throughout training. λ × 0.005 = 5e-5 vs main loss ~6, so the penalty had no measurable effect.

| step | exp361 (contrast λ=0.01) | exp353 (no contrast) | Δ |
|---|---|---|---|
| 1000 | 1.8815 | 1.8801 | +0.0014 |
| 1400 | 1.8329 | 1.8285 | +0.0044 |
| 1800 | 1.7971 | 1.7964 | +0.0007 |
| 2400 | 1.7590 | 1.7575 | +0.0015 |

**Pattern**: indistinguishable from exp353 (within ±0.005 eval noise). The penalty was effectively dormant.

**Why**: with small-std init (0.001), tables are already near-orthogonal, so the contrast gradient is tiny and easily dominated by the main loss. To have an effect would need λ ~100× larger, but that would likely just push tables into pathological non-orthogonality without helping accuracy.

## Bigger conclusion

Cumulative pattern across exp353–exp361 (LR, β1, Lookahead, hard-mining, per-row LR scale, soft-weight-grad, inter-table contrast): **none of the cheap LUT-specific or optimizer-side tricks beats baseline by more than ±0.01 bpb**. The bs-scaling effect (−0.11 to −0.19 bpb) is fundamentally about more gradient samples per row, and we have not found a way to manufacture that without actually computing more gradients.

**Open paths** still worth trying:
- Multi-token prediction (Medusa-style): orthogonal benefit, lifts both vanilla and LUT.
- Inter-ROW contrastive (within table, not inter-table): more directly targets row-collapse problem.
- Per-row replay buffer with custom optimizer (proper sparse-aware Adam).
- Lifting back to bs=48 or bs=96 and exploring widening (LUT shape sweep at higher batch).
