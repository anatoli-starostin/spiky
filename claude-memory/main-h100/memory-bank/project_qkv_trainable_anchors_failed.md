---
name: project_qkv_trainable_anchors_failed
description: exp568 — SoftAnchorPair qk_lut + τ-anneal-then-hard-snap LOSES to random anchors at E=96 by +0.012 bpb; trainable-anchor hypothesis refuted at this scale
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Test:** exp567 fork (E=96, 1.4768 @ 103.02M) with the ONLY change being qk_lut: TinyMHLut(HARD argmax, NAP=4, tph=256, random anchors) → `SoftAnchorPairMHLUT` (directed-gradient soft anchor pairs, NAP=4, tph=256, **learnable** anchors). Other LUTs (v_lut, out_proj, residual_lut) kept on TinyMHLut. Schedule: `anchor_tau` cosine 1.0 → **0.001** over steps 0–3000, then `qk_lut.hard = True` for exact argmax forward through step 8000. anchor_logits (7.08M params) → AdamW (lr=3e-4, no wd); LUT weights still on Lion. Total 110.10M (+7.08M anchor_logits).

**Hypothesis tested:** at E=96, random NAP=4 anchors have C(96,4)≈3.3M combinations available — 5× sparser than E=64's C(64,4)≈635k. So fixed-random anchors might be missing informative input dims; learning anchor coordinate pairs could outperform random.

**Result: clear regression. exp568 = 1.4884 @ 8K, vs exp567 1.4768 = +0.0116, vs exp513 1.4825 = +0.0059** (also worse than the E=64 baseline). Net per-param cost: +0.0016 bpb/M — same order of magnitude as exp566's failed big-head MLP.

**Curve shape:**

| step | exp568 | exp567 | Δ |
|--|--|--|--|
| 800 | 1.8246 | 1.8275 | **−0.003** (peak lead, soft mode early) |
| 2000 | 1.6632 | 1.6622 | +0.001 (flip — anneal biting) |
| 2800 | 1.6146 | ~1.6113 | +0.003 (last pre-snap eval, τ≈0.012) |
| **3000 (SNAP)** | **1.6028** | 1.5986 | +0.004 |
| 4000 | 1.5628 | 1.5554 | +0.007 |
| 5000 | 1.5302 | 1.5229 | +0.007 |
| 6000 | 1.5091 | 1.4995 | +0.010 |
| 7000 | 1.4961 | 1.4853 | +0.011 |
| **8000** | **1.4884** | **1.4768** | **+0.012** |

**The snap was clean (good)**: 200-step Δ at the boundary (1.6146 → 1.6028) is normal training progress with no upward blip. τ_final=0.001 was sharp enough that soft mode at step 2800 was essentially already hard. **So the regression is NOT a snap-discontinuity artifact** — it's the trainable-anchor mechanism itself losing.

**Three causal candidates** (most likely #1 + #2):

1. **Random NAP=4 anchors are near-optimal already at this scale.** The 5× sparsity argument was too pessimistic — at NAP=4 the LUT has only 16 rows to populate, and any 4 of 96 input dims is roughly as informative as any other for q,k routing. The dominance of one specific subset is small.

2. **Soft-mode LUT weight co-adaptation, then snap mode-mismatch.** During steps 0–3000 the LUT `weights` co-adapted to *soft anchor blends* (a continuous mixture of input coords). Post-snap the forward uses hard argmax (a single coord); the weights are now tuned for the wrong forward and have to re-fit. Slope evidence: post-snap exp568 descended at the same rate as exp567 (no catch-up), so the soft phase didn't pre-fit anything reusable.

3. **Vanishing-gradient on anchor_logits as τ → 0.001.** The softmax over 96 input dims at τ=0.001 produces effectively-one-hot outputs; the derivative through softmax becomes near-zero. So `anchor_logits` is gradient-starved late in the anneal, and might just freeze at whatever (random-ish) state they reached at τ≈0.1. Less likely than #1/#2 given the magnitude.

**What's preserved from this run:**
- **Mechanism is sound when isolated**: SoftAnchorPair forward + anchor_tau scheduling + `.hard=True` snap all worked (gradient flow, no NaNs, no diverging spikes, snap discontinuity ~zero).
- **τ_final=0.001 is the right snap target** — sharper than the 0.01 default; if used in a future test, keep 0.001 or sharper.
- **Anchor params should be on AdamW, not Lion** — the `name.endswith('.anchor_logits')` filter in the param-grouping code is the correct dispatch rule (Lion's sign-momentum on softmax logits would be wrong).

**Cross-experiment context (capacity-bloat per-param efficiency on exp513/567 class):**

| change | source | Δparams | Δbpb | bpb/M |
|--|--|--|--|--|
| D=384→512 (exp562) | exp513 | +10.5M | −0.0111 | −0.00106/M |
| residual_tph 128→256 (exp530) | exp513 | +18.9M | −0.0094 | −0.00050/M |
| E=64→96 (exp567) | exp513 | +13.6M | −0.0057 | −0.00042/M |
| **MLP head (exp566)** | **exp513** | **+21.4M** | **+0.0066** | **net drag** |
| **qkv trainable anchors (exp568)** | **exp567** | **+7.1M** | **+0.0116** | **net drag** |

Two architectural perturbations in the exp56x series both lost (exp566 head, exp568 anchors); the productive levers stay D and residual_tph (raw bus width).

**Closed direction:** trainable anchors as a drop-in replacement for random anchors at small scale on the exp513/567 backbone. **Open:**
- Could be re-tried at much larger scale (E=256+, NAP=8+, harder routing problem) where random anchor sparsity is genuinely binding.
- Or with a fundamentally different schedule (no snap at all — train soft all the way; or co-train LUT weights only after anchors converge; or freeze anchors very early at random init and only learn weights).
- For the matmul-free LM at the current 100M scale, **stay with random fixed anchors (TinyMHLut)**.

Code at `/home/starost/spiky/nanochat_exps/exp568_qkv_anchored/`; mechanism re-usable for future tests (the τ-schedule + hard-snap pattern works).
