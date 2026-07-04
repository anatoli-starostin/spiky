---
name: project-nap6-out-proj-sweet-spot-at-e96
description: exp569 + exp570 (both killed early) confirm NAP=6 is the sweet spot for out_proj at mid-scale (exp567/E=96/~100M); NAP=4 wins only at tiny scale where row-collapse bites
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Two killed experiments on the exp567 (E=96, 1.4768 @ 103.02M, 475 Mbits/tok) baseline, both targeting the out_proj NAP-shape question:**

| exp | change | params | bandwidth | wall-clock | last-eval bpb (step) | killed at |
|--|--|--|--|--|--|--|
| **exp569** | out_proj → TinyMultiNapMultiHeadLut summed over `[(4,128),(6,64),(8,96)]` (exp387 recipe port) | **82.97M** (−20.05M) | **461 Mbits** (−13.6 Mbits) | ~same as exp567 | 1.6086 @ step 3000 (+0.010 vs exp567) | step 3000, monotonic widening |
| **exp570** | out_proj NAP=6/tph=1024 → **NAP=4/tph=2048** (exp390 recipe port) | 84.15M (−18.87M) | **493 Mbits** (+18.8 Mbits) | **+55%** (0.45 vs 0.29 s/step) | 1.6026 @ step 3000 (+0.004 vs exp567) | step 3000, after a brief +0.0017 narrowing at step 2000 reversed back |

**Both LOST to exp567** on bpb. exp570 was triply bad: also more bandwidth AND slower wall-clock.

**Conclusion: NAP=6 is the sweet spot for out_proj at exp567-class mid-scale (~100M, E=96).** User's hypothesis confirmed: NAP-shape rules are scale-conditional. Specifically:
- **qk_lut**: NAP=4 sweet spot (hard-argmax stability, finalized by exp508-513 sweep).
- **v_lut, out_proj, residual_lut**: NAP=6 sweet spot at mid-scale (all three already there in exp513/567).

**Mechanism reconciliation with prior tiny-LUT-LM findings** ([[project_nap4_only_out_proj_sota]], [[project_multinap_out_proj_sota]]):
- exp390 (NAP=4 out_proj tph=2048, **tiny 43M backbone**) won −0.021 vs baseline at bs=16. There the row-collapse pathology bit: NAP=6/8 had ~128 tokens/row/step (too sparse for clean Adam stats), NAP=4 had ~512 tokens/row/step (dense gradient coverage).
- At exp567 mid-scale, the bigger backbone gives the LUT routing enough capacity that NAP=6 stays gradient-covered. The "row-collapse" doom-floor of tiny scale is gone, and NAP=6's deeper table (64 rows vs 16) becomes a real capacity win that NAP=4 can't recover from with more tables.
- **Critical implication**: don't blindly port tiny-LUT-LM recipes (exp386–exp400 era) to mid-scale. Many were targeted fixes for problems mid-scale doesn't have.

**Wall-clock-vs-bandwidth tradeoff (new datapoint from exp570):** doubling table count (1024 → 2048) at fixed NAP-depth roughly **doubled out_proj bandwidth AND wall-clock per-step**. On H100 (dense-matmul optimized) the linear relationship from bandwidth to wall-clock holds — sparse gathers don't get the architecture's discount. Spending bandwidth via tph (more tables) is also a wall-clock cost.

**Closed direction at mid-scale:** out_proj NAP-shape perturbations away from NAP=6 (both NAP=4 dense and multi-NAP sum). exp567 baseline IS the right NAP shape for out_proj. Next levers should preserve NAP shapes and look elsewhere: residual_tph widening (proven on E=64, untested on E=96), joint E×D, per-layer schedules, noise_eps, more layers.

Files: `/home/starost/spiky/nanochat_exps/exp569_multinap_out/`, `/home/starost/spiky/nanochat_exps/exp570_out_nap4_tph2048/` (both retain metrics.csv up to kill point; no checkpoints).
