---
name: project_head_arch_doesnt_stack
description: "exp575-583 arc: tied dot head + Linear head (over D-stream from residual_lut) do NOT stack — both extract overlapping information; dual-head + per-layer residual_lut caps ~+0.04 bpb above exp567"
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Arc target:** can we beat exp567 (full LUT-LM arch at E=96, 1.4768 @ 103M / 475 Mbits) by widening E to 192 and/or adding a parallel head topology?

**Sequence of architecture variants at E=192 / d_v=32 / D=192 (H·d_v=E invariant):**

| exp | residual_lut placement | head topology | final bpb | params | trunk Mbits | total Mbits |
|--|--|--|--|--|--|--|
| exp575 | NONE (no D-stream) | tied dot only (E-stream) | (proj ~1.54, killed @6K = 1.5754) | 124M | 87 | 289 |
| exp582 | 1 final (post-layer-6) | tied dot + Linear(D, V) DUAL | (proj ~1.53, killed @3K = 1.6463) | 132M | 88 | 491 |
| **exp583** | **per-layer × 6** | **tied dot + Linear(D, V) DUAL** | **1.5200 @ 8K** | **140M** | 96 | 499 |
| exp567 (E=96 ref) | per-layer × 6 | Linear-only UNTIED, no tied dot | **1.4768 @ 8K** | 103M | 72 | 475 |

**Locked findings:**

1. **The two heads (tied dot on E-stream + Linear on D-stream) DO NOT STACK.** They extract overlapping next-token signal; once one head learns to predict well, the other adds essentially nothing. Removing one head (exp575: tied only, no Linear; or exp567: Linear only, no tied dot) gives strictly better quality at lower bandwidth than the dual.

2. **Adding per-layer residual_lut on top of dual-head buys ~5 mb vs single-final.** exp582 vs exp583 = 0.005-0.010 difference throughout training — six per-layer LUTs vs one final LUT in the residual pathway is nearly inert. The D-stream's signal is largely captured by ONE good projection; iterative refinement across layers doesn't compound when a tied dot head also reads from the E-stream.

3. **Dual-head architecture caps at +0.04-0.05 bpb above exp567** at E=192, regardless of residual_lut placement (1 final or per-layer × 6). The cap isn't because residual_lut capacity is insufficient — it's because the dual-head topology is suboptimal vs single-head.

4. **The "minimal arch + tied dot" path (exp574/575) and "full arch + Linear head" path (exp567) are alternative architectures that target different bandwidth-quality points.** Mixing them costs both.

**Mechanism hypothesis:** with two heads writing the same target distribution, gradient is split between them. Both heads learn the gross structure quickly (predict frequent tokens, capture syntactic regularities), but neither becomes specialized in the long tail. A single head with more capacity (exp567's Linear(D=384, V) is bigger than exp583's Linear(D=192, V) + tied dot(E=192, V)) probably ends up with better calibrated logits on rare tokens.

**Strategic implication:**
- For "match exp567 at lower bandwidth": stick with the tied-dot-only family (exp574/575); accept ~+0.05 above exp567.
- For "beat exp567 absolute": try exp567's architecture (Linear-only head, no tied dot) at WIDER E (E=192 or E=384). Single-head with bigger E-stream, no dual-head dilution.
- The "dual head" direction is closed at this scale.

**Vs vanilla (exp328 = 1.3882):**
- exp583 gap: +0.132
- exp567 gap: +0.089
- exp575 (proj): +0.15
- Per-token convergence to vanilla still bounded below by exp567's +0.089 gap.

## Update 2026-05-26 — exp584 (dual UNTIED head at E=96)

**exp584 = exp567 + parallel UNTIED Linear(E=96, V=32768) head + LayerNorm(E)**, alongside the existing Linear(D=384, V). Goal: test whether separate-weights (untied) avoids the tied-dot overlap problem from exp582/583. Final: **1.4795 @ 106.17M / 575 Mbits total** vs exp567 1.4768 @ 103M / 475 Mbits.

**Statistical tie (+0.003 bpb, +3M params, +100 Mbits head bandwidth) — UNTIED dual head is neither a win nor a loss vs single Linear at E=96.** The two heads can learn independent directions in principle, but at E=96 there's no useful "second direction" beyond what Linear(D, V) already captures. The D-stream activation already contains the predictive structure; reading the E-stream separately doesn't surface anything new.

Combined hierarchy at E=96:
- exp567 (Linear D only): **1.4768** — current ceiling
- **exp584 (Linear D + UNTIED Linear E): 1.4795** — wash, no gain
- exp582 (Linear D + TIED dot E, 1 final resid): proj ~1.53 — TIED hurts
- exp583 (Linear D + TIED dot E, per-layer resid): **1.5200** — TIED hurts more

**Refined conclusion: at E=96 the LUT-LM final activation is SATURATED by a single Linear head. Any second head either fails (tied → +0.05) or no-ops (untied → +0.003). Head topology variations exhausted at this scale.** Levers to beat exp567 must target: (1) wider D (D=512+, exp562/565 path); (2) wider E (E=192+ at full arch, not tied-only stripping); (3) LUT routing improvements (gradient/coverage); (4) sparser trunk to free bandwidth budget. NOT head topology.

Files: `/home/starost/spiky/nanochat_exps/exp575_E192_dv32/`, `/exp582_e192_dual_head/`, `/exp583_e192_perlayer_resid/`, `/exp584_e96_dual_untied/`.
