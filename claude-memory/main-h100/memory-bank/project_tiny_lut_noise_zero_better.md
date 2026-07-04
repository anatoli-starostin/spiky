---
name: tiny-lut-noise-zero-better
description: "At tiny LUT-LM scale, argmax_noise_eps=0.0 beats =0.002 by ~0.006 bpb. Opposite sign from the larger-scale finding where noise=0.002 helped TinyMHLut(soft) match SoftMHLut(hard=True)."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Tiny LUT-LM: noise=0 beats noise=0.002 (2026-05-15, exp352)

exp352 = exp340 fork with `argmax_noise_eps` 0.002 → 0.0. All else identical (E=48, out_tph=128 uniform, all-soft backward, bs=16, 8K steps).

| step | exp340 (noise=0.002) | exp352 (noise=0.0) | Δ (no-noise − noise) |
|---|---|---|---|
| 200  | 2.3322 | 2.3181 | −0.0141 |
| 1000 | 1.8972 | 1.8929 | −0.0043 |
| 2000 | 1.7924 | 1.7901 | −0.0023 |
| 4000 | 1.6997 | 1.6988 | −0.0009 |
| 6000 | 1.6536 | 1.6499 | −0.0037 |
| 8000 | **1.6366** | **1.6302** | **−0.0064** |

No-noise was ahead from step 200 onward; gap widens slowly from −0.001 to −0.006 by step 8000.

**Sign-flip from larger-scale finding**: at the exp257/exp260 scale (LUT-LM at 300M+ params), noise=0.002 was needed for TinyMHLut(soft) to match SoftMHLut(hard=True) — bf16 was doing implicit regularization there. At tiny scale (43 M params, much smaller LUTs) the soft path is **self-regularizing enough**; the noise perturbation is just damage.

**How to apply:**
- Tiny LUT-LM forks (43 M-ish, E=48 ish): use `argmax_noise_eps=0.0` as default.
- Larger LUT-LM forks (300 M+): keep `argmax_noise_eps=0.002` per the [[soft-lut-noise-regularization]] finding.
- Crossover scale unknown — worth re-testing at exp326 scale (620 M) if revisited.

**Effect on tiny LUT-LM SOTA**:
- exp340 baseline: 1.6366 (with noise)
- exp352 new tiny baseline: **1.6302** (no noise)
- Best tiny LUT-LM stays exp349 = **1.4478** (bs=96 + noise=0.002). bs=96 + noise=0.0 not yet tested.
