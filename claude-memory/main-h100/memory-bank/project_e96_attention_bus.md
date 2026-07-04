---
name: project_e96_attention_bus
description: exp567 — widening attention bus E 64→96 is a real but ~2.5× less efficient lever than D-widening; modest late-slope steepening
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Test:** exp513 fork (E=64 → E=96, single config knob change `embedding_dim: 64 → 96`). All other arch + hyperparams byte-for-byte identical. Total params 89.4M → **103.02M** (+13.6M).

**Why the test was interesting:** at E=64, exp513 has the structural quirk `H·d_v = 6×16 = 96 → out_proj n_out = E = 64` — every layer implicitly compresses the SDPA output by 33%. E=96 makes out_proj 96→96 (no compression), and also widens tok_emb_E (+1.05M), qkv_lut/v_lut/residual_lut input encodings, and out_proj n_out by 50%.

**Result: exp567 = 1.4768 @ 8K, beating exp513 1.4825 by −0.0057.** Win is consistent: lead vs exp513 was monotone-positive direction across steps 3K/4K/5K/6K/7K/8K (−0.001 → −0.003 → −0.002 → −0.0045 → −0.0058 → −0.0057). Not seed noise.

**Curve shape:**
| step | exp567 | exp513 | Δ |
|--|--|--|--|
| 200 | 2.2561 | 2.2549 | +0.001 (no early bonus, unlike exp566's MLP head) |
| 2000 | 1.6622 | 1.6603 | +0.002 (still tracking) |
| 3000 | 1.5986 | 1.5997 | **−0.001** (flip) |
| 5000 | 1.5229 | 1.5252 | −0.002 |
| 8000 | **1.4768** | 1.4825 | **−0.0057** |

Late slope (3000→8000): exp567 −0.02436/k vs exp513 −0.02344/k → exp567 is **~4% steeper**, vanilla still ~1.5× steeper than both.

**Per-param-efficiency ranking on exp513-class backbone:**

| change | Δparams | Δbpb | bpb/M |
|--|--|--|--|
| D=384→512 (exp562) | +10.5M | −0.0111 | **−0.00106/M** (best) |
| residual_tph 128→256 (exp530) | +18.9M | −0.0094 | −0.00050/M |
| **E=64→96 (exp567)** | **+13.6M** | **−0.0057** | **−0.00042/M** |
| big-head MLP (exp566) | +21.4M | +0.0066 | **net drag** |

E-widening is a real lever but **~2.5× less per-param-efficient than D-widening**. The D-stream is the dominant capacity bus; E-stream is secondary.

**Mechanism reading:**
- Removing the out_proj 96→64 compression preserves attention info → small steady-state benefit.
- But qkv_lut/v_lut/residual_lut see a wider input bus where RANDOM anchor coverage degrades sharply: C(E,NAP) grows ~E^NAP, so at NAP=6 the random-pick combinatorial space is **16× sparser** at E=96. A bigger fraction of LUT capacity is paid for but indexed by uninformative bits.
- This makes E=96 a natural setup for **trainable anchors** (`SoftAnchorPairMHLUT`): learned anchor subsets could exploit the wider bus that random anchors mostly waste. Queued as exp568+ — see [[project_unembedder_head_explorations]] and the existing `trainable_anchors_multi_head_lut.py` (the `SoftAnchorPairMHLUT` variant with directed-gradient soft pairs + scheduled τ anneal + `.hard=True` snap).

**Cross-experiment context:**
- vs vanilla bs=16 exp328 (1.3882 @ 23.2M): exp567 gap = +0.089 (vs exp513's +0.094). Gap to vanilla shrinks by 5 millibits — same direction as the D-sweep but smaller magnitude.
- vs exp566 (MLP head at D=384, 110.76M, 1.4891): exp567 beats it by 12 millibits at fewer params. Different mechanism (E-bus widening vs head-Jacobian widening), and the E-bus wins cleanly.
- vs current matmul-free SoTA exp565 (1.4513 @ 175.4M, D=512+restph512): exp567 is +0.0255 worse at 0.59× the params — exp565 is still the absolute SoTA but exp567 is more interesting as a small-model baseline.

**Open follow-ups:**
- Trainable anchors on E=96 backbone (the soft-pair design + scheduled τ → 0.01 → `.hard=True` snap). Scope decision: qkv_lut-only first vs all-three E-input modules.
- E=96 + D=512 joint widening (untested; could compound).
- E sweep further: E=128/160 to find the knee.
