---
name: project_soft_winner_dead_end
description: "Scaled-hard LUT forward (out = softmax_winner_coeff * W[winner]) is a dead end — coeff attenuation creates a pathological landscape where T_sel drifts up and the model diverges."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

exp485 (2026-05-22): user idea via Telegram — "soft forward but only for the winner.
Like argmax but weights multiplied by softmax winner coeff." Implemented as new
`backward_mode='soft_winner'` in `tiny_multi_head_lut.py`: forward picks argmax row
(winner) but scales its weights by the winner's softmax coefficient
`coeff = softmax(ts/T_sel)[winner]` (= max of selection softmax, scalar in (1/K,1] per
table). `out = coeff * W[winner]`. Single-row inference (same bandwidth as ste/soft),
deterministic (train==eval), coeff is a differentiable confidence gate giving an
x-gradient path through the selection scores. (`_soft_lut_fwd_body_winner`,
`_soft_lut_bwd_body_winner`, `_TinyMHLutSoftWinner`. Analytic backward verified vs
autograd reference to ~1e-15.)

**FAILED — diverges.** Forked exp475 (1.4962), gap widened monotonically: step 200
−0.001 → 400 +0.051 → 600 +0.099 → 800 +0.140 → 1000 +0.204, and val bpb went UP
800→1000 (1.978→1.995). Killed at step 2000 (1.9491 vs exp475 1.6748, +0.274).

**Root cause (diagnosed live from temperatures.csv):** at init the softmax over K=64
rows is near-uniform → winner coeff ≈ 1/K ≈ 0.016 → output crushed ~64×. To recover
magnitude the model would need to SHARPEN selection (lower T_sel) or scale weights up.
Instead the learnable T_sel drifted the WRONG way — UP (0.50→0.56 by step 800), making
the softmax MORE diffuse → coeff SMALLER → more attenuation → worse loss. exp475's T_sel
moves DOWN (0.50→0.49) over the same window. The coeff-scaling couples output magnitude
to selection confidence in a way that yields a bad optimizer equilibrium and a runaway
attenuation feedback loop.

**Conclusion:** don't multiply LUT output by selection confidence. If a confidence gate
is wanted, it must NOT be the only thing controlling output magnitude (decouple), or the
coeff must be normalized so init magnitude ≈ argmax. Argmax-forward + soft-backward
(exp475) stays the default. Joins [[project_prob_forward_dead_end]] (the other
exp475-fork forward-modification dead end this session). Code left in place, off by
default (`backward_mode='soft_winner'`).
