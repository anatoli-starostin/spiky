---
name: project_big_head_mlp_falsified
description: "exp566 — MLP head (Linear→1024→GELU→Linear) at fixed D=384 FAILS to steepen late slope; D-sweep gain was wider residual bus, not head Jacobian"
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Test:** isolated big-head MLP on exp513 backbone (D=384 byte-for-byte identical), changing ONLY the unembedder from `Linear(384, V=32768)` (12.58M) to `Linear(384,1024) → GELU → Linear(1024, V)` (33.95M). Total 110.76M vs exp513's 89.4M.

**Hypothesis tested (user, 2026-05-25):** "bigger head propagates better gradients to downstream parts of the model" — would manifest as a STEEPER late slope (gap to vanilla narrowing through steps 3000–8000), not just a lower floor.

**Result: hypothesis REFUTED at this scale.** Final 1.4891 @ 8K — **worse than exp513 1.4825 by +0.0066** at +21.4M params.

| step | exp566 | exp513 | vanilla328 | Δ(566−513) |
|--|--|--|--|--|
| 200  | 2.1572 | 2.2549 | 2.3437 | −0.098 (strong early lead) |
| 1000 | 1.7597 | 1.7803 | 1.8217 | −0.021 |
| 2000 | 1.6478 | 1.6603 | 1.6659 | −0.013 |
| 4000 | 1.5551 | 1.5583 | 1.4950 | −0.003 |
| 5000 | 1.5280 | 1.5252 | 1.4442 | **+0.003** (crosses behind) |
| 8000 | 1.4891 | 1.4825 | 1.3882 | **+0.007** (net drag) |

Late-slope (3000→8000): exp566 **−0.0205/k**, exp513 **−0.0234/k**, vanilla **−0.0375/k**. exp566 descends SLOWER than exp513 — the MLP head is not propagating better gradients; it gave a one-shot early offset (−0.10 at step 200) that LR-anneal washes out monotonically.

**Reframes the D-sweep finding ([[project_lut_lm_residual_width_sota]]):** exp562 (D=384→512, +21.4M) won by −0.0111 NOT because the head Jacobian got richer — exp566 isolated and falsified that. The mechanism is the **wider residual bus**: residual_dim, residual_lut output width, ln_final, AND head input width all grew together; residual_lut writes more dims/token into the residual stream; the same Linear head reads from a higher-bandwidth representation. Consistent with the prior wide-beats-deep result for residual_lut.

**Implication for the per-token gap to vanilla ([[project_lut_lm_pertoken_gap_vs_vanilla]]):** head architecture is NOT the lever for late-slope deficit. Next probes should target:
- Gradient resolution in the LUT routing path (denser per-row gradients, hardness schedules)
- Wider residual bus + bigger head jointly (untested — could compound)
- Smooth auxiliary head/branch that fades (training-time gradient teacher, not inference path)

**Open caveats:** 33.9M head is a modest perturbation (~31% of model); deeper/wider MLP (hidden=4096+) or D=512+MLP head jointly remain untested. But within the budget the answer is clear: head depth/non-linearity at fixed D is inert-to-net-negative.

Code at `/home/starost/spiky/nanochat_exps/exp566_big_head_mlp/` — config copy of exp513 with single edit at `train.py:260-269` (Sequential MLP). Same param grouping (both new Linears go to AdamW decay group via the ndim==2 branch).
