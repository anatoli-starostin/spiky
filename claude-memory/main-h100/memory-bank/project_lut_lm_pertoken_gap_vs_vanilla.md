---
name: project_lut_lm_pertoken_gap_vs_vanilla
description: "LUT-LM vs vanilla per-token convergence — crossover at ~step 3000, late-slope deficit; the real research target. Use exp476 (untied head) as the fair reference, not exp328 (tied)."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**The actual research goal (user, 2026-05-25):** reach **bpb-per-token convergence efficiency on par with vanilla** — the LUT-LM learning curve (bpb vs tokens at fixed budget) should match a vanilla transformer's. Training longer or adding params/tables is OFF-target: both just slide models down their own curves; neither closes the *gap between* curves. Stop optimizing absolute bpb via bloat/steps.

**Fair-comparison reference is `exp476` (untied head), NOT `exp328` (tied head).** The LUT-LM is **structurally forced into untied** (its head reads D=384 stream, embedding writes E=64/96 stream → can't share weight matrix); for apples-to-apples, vanilla must be untied too. Untying *hurts* vanilla (exp328 tied 1.3882 → exp476 untied 1.4143, +0.026 bpb at +12.6M params) because the tied matrix trains under both input-embedding and output-head gradients — more signal per param. **Some of the "LUT-LM vanilla gap" is the mandatory-untied structural cost, not a LUT routing issue.**

| reference | head | params | bpb@8K | role |
|--|--|--|--|--|
| exp328 | tied | 23.2M | 1.3882 | absolute floor; NOT fair to compare LUT-LM to |
| **exp476** | **untied (single-knob fork of exp328)** | **35.79M** | **1.4143** | **the fair vanilla reference** |
| exp567 | untied LUT-LM (E=96, D=384) | 103.0M | 1.4768 | current cleanest small LUT-LM baseline |
| exp513 | untied LUT-LM (E=64, D=384) | 89.4M | 1.4825 | prior LUT-LM ref before E-sweep |

**The gap has a shape — inverts at ~step 3000** (exp567 LUT vs exp476 vanilla-untied, both bs=16/8K/RoPE/8192-tok):

| step | exp567 (LUT) | exp476 (vanilla-untied) | gap (LUT − vanilla) |
|--|--|--|--|
| 200 | 2.2561 | 2.3331 | **−0.077** (LUT ahead) |
| 800 | 1.8275 | 1.8815 | −0.054 |
| 1000 | 1.7820 | 1.8282 | −0.046 |
| 2000 | 1.6622 | 1.6698 | −0.008 |
| **3000** | 1.5986 | 1.5838 | **+0.015 (crossover)** |
| 4000 | 1.5554 | 1.5248 | +0.031 |
| 5000 | 1.5229 | 1.4737 | +0.049 |
| 6000 | 1.4995 | 1.4429 | +0.057 |
| 7000 | 1.4853 | 1.4250 | +0.060 |
| 8000 | **1.4768** | **1.4143** | **+0.063** |

- **LUT-LM is MORE token-efficient than vanilla-untied for the first ~3000 steps** (big discrete tables grab common structure fast, even beating a ~3×-smaller vanilla), then crosses over and vanilla pulls away.
- The deficit is entirely a **late-training slope** problem. Second-half descent (steps 3000→8000): vanilla-untied **−0.0339/k** vs LUT **−0.0244/k** (~1.39× steeper for vanilla, vs 1.50× when comparing to tied).
- Mechanism hypothesis: discrete argmax routing has limited resolution + sparse/noisy per-row gradients. Early gross structure is easy (noise irrelevant); late signal is subtle (rare tokens, fine distinctions) and gets drowned by routing noise — vanilla's smooth weights have no such floor. Consistent with the prior "bigger batch = denser per-row gradients" lever.
- **Strategy:** target the late slope (gradient resolution / per-row signal density / hardness schedules / smooth auxiliary path that fades), NOT more tables, NOT more steps. We already beat vanilla early.

**Vs tied vanilla (kept for reference only, NOT the fair comparison):** exp567 vs exp328 = +0.089 (vs +0.063 vs untied) — 2.6 mb of that gap is the mandatory-untied structural cost, not LUT.

**Cosine-tail red herring (ruled out):** the "model keeps gaining to the very end" is NOT special — every model here, **dense vanilla GPTs included**, puts ~25–33% of post-midpoint loss drop into the final quarter (it's the shared cosine LR anneal). Per-module `weight_deltas.csv` (logged every 200 steps in these runs): all modules shrink their update magnitude by the identical ~0.16–0.18 from step 4000→8000 (= the global lr_scale decay). By size-normalized `rel_delta` the **unembedder freezes FASTEST** (0.16), while the LUT backbone stays most active late (out_proj 0.28, qk_lut 0.23, residual_lut 0.21). So late learning lives in the LUT backbone, not the head — refuted the "anneal hits the head" idea. See [[project_lut_lm_residual_width_sota]].

**What's been ruled out as late-slope levers** (all on exp513/567 backbone):
- Bigger MLP head ([[project_big_head_mlp_falsified]]): exp566 = +0.012 vs exp513, no slope steepening, gradient-propagation hypothesis refuted.
- Trainable qkv anchors ([[project_qkv_trainable_anchors_failed]]): exp568 = +0.012 vs exp567, random NAP=4 anchors near-optimal at this scale.
- Big residual/D bloat ([[project_lut_lm_residual_width_sota]]): closes absolute bpb via params, but per-token gap to vanilla unchanged (off-target).

**Still open:** gradient-resolution / per-row signal density / hardness schedules / smooth auxiliary path that fades during training.
