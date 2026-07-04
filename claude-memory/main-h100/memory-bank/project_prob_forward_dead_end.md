---
name: project_prob_forward_dead_end
description: Probabilistic-forward LUT selection (sample one row from softmax instead of argmax during training) is a dead end — +0.05 bpb worse than argmax and 2.7x slower.
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

exp484 (2026-05-22): forked exp475 (TinyMHLut soft/argmax, 1.4962 @ 8K) with a new
`backward_mode='prob'` on all 4 LUT modules. **Final = 1.5474, Δ=+0.0512 vs exp475** —
clean structural loss.

**Mechanism (`backward_mode='prob'` in `tiny_multi_head_lut.py`):**
- TRAIN forward: per (batch, table), sample ONE row from `softmax(ts/T_sel)` via
  `torch.multinomial` (instead of argmax). Single-row selection → same inference
  bandwidth as soft/ste. (`_soft_lut_fwd_body_prob`, `_TinyMHLutProb`)
- EVAL forward: deterministic argmax (`_soft_lut_fwd_body_einsum`) so `[VAL]` is
  reproducible & comparable to exp475.
- Weight gradient: hard `index_add_` at the SAMPLED row (STE-style).
- Input gradient: derived from the FULL softmax distribution (`_soft_lut_bwd_body_prob`,
  uses actual `p=d/(T+|d|)`, NOT reconstructed from sampled-index bits) — the gradient
  of E[output] w.r.t. input doesn't depend on which row was sampled. (User insisted on
  this: "input gradient should be derived from softmax distribution, selected index is
  irrelevant for input gradients.")

**Result:** gap was DEAD FLAT at +0.05 bpb across all 40 evals (steps 200→8000), vs
exp475's actual trajectory. Not an early-phase tax — a constant structural deficit.
Hypothesis was that sampling non-optimal rows during training improves cold-row
gradient coverage at bs=16; it didn't materialize. The forward noise from picking
sub-optimal rows just degrades the learned function with no recovery.

**Also 2.7x slower**: 1.104 h vs exp475's 0.408 h. `torch.multinomial` is a dynamic op
(not `@torch.compile`-able), and the prob backward body runs uncompiled.

**Conclusion:** argmax-forward + soft-backward (exp475 recipe) stays the default. Don't
revisit stochastic forward selection. Related dead ends: soft weight gradients grafted
on hard forward [[project_soft_wgrad_and_contrast_dead_ends]], windowed-grad smoothing
[[project_windowed_grad_dead_end]], hard mining [[project_hard_mining_tiny_lut]] — the
whole "improve gradient coverage at fixed phys-batch" family caps out negative. Real
lever remains true batch scaling / grad_accum [[project_grad_accum_reproduces_big_batch]].
Code left in place (off by default; only active with `backward_mode='prob'`).
