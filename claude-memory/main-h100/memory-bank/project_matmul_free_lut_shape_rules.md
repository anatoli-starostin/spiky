---
name: project_matmul_free_lut_shape_rules
description: "Matmul-free LUT-LM (qk hard-argmax line, exp511-532): per-module nap-depth rule (qk wants wide-shallow nap=4, decoders v_lut/residual want deep nap=6), d_v helps monotonically, residual_lut is the most capacity-responsive module. SOTA exp530=1.4731. Hard v-branch hurts."
metadata:
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Matmul-free LUT-LM shape sweep (2026-05-24), E=64/D=384/6L, bs=16, 8000 steps, untied Linear(D,V) head.** All "matmul-free" = the 4 per-layer LUT *projections* (qk, v, out, residual) are hard-argmax `TinyMHLut` lookups (no matmul); attention SDPA and the Linear unembedder are still real matmuls. Base = exp475 (all-argmax ref, 1.4962).

## qk hard-argmax nap sweep (fixed 4096 entries/table, param-matched)
qk_lut reshaped at constant `2^nap * tph = 4096`, q,k only (no v-branch):
- nap=2/tph=1024 (exp511) = 1.4862
- nap=3/tph=512 (exp512) = 1.4888
- **nap=4/tph=256 (exp513) = 1.4825 ← sweet spot**
- nap=5/tph=128 (exp519) = 1.4880
- (soft nap=6/tph=64, exp508 = 1.4840)
**qk prefers WIDE-SHALLOW (nap=4); non-monotonic with nap=4 the clear optimum, and it BEATS the soft qk, fully matmul-free.**

## v_lut and residual_lut prefer DEEP (nap=6) — opposite of qk
- v_lut nap sweep: nap=4/tph=1024 (exp524) = 1.5055 (much worse); nap=6/tph=320 (exp513) = 1.4825 (best); nap=8/tph=128 (exp525, +7M) = 1.4843 (~tied, inert). v_lut also wants MORE tph (tph=320 > tph=256 exp526=1.4876 by ~5mb).
- residual_lut nap (PARAM-MATCHED, 16384 entries/table, both 108.3M): nap=4/tph=1024 (exp532) = 1.4787 vs nap=6/tph=256 (exp530) = 1.4731 → deep wins by ~6mb. (exp531 nap=4/tph=256, smaller, trailed ~12mb.)
**RULE: the dominance-based attention-input projection (qk) wants many shallow tables (nap=4); the value/residual DECODERS (E->d_v, E->D) want fewer deeper tables (nap=6).**

## d_v (value dim) helps monotonically
At v_lut nap=6/tph=256: d_v=12 (exp528) = 1.4943, d_v=16 (exp526) = 1.4876, d_v=24 (exp527) = 1.4812. ~−0.006 per step, ~+2.36M params/step. The default d_v=16 was a mild value-bandwidth bottleneck. (d_qk stays 64, RoPE unaffected.)

## residual_lut is the most capacity-responsive module — SOTA
residual_lut tph 128->256 (exp530, +18.87M -> 108.3M) = **1.4731**, −0.0094 over exp513 — the biggest single lever and ~2x the seed-noise band (clearest signal). v_lut tph 256->512 (exp529, +14M) only −0.0026 (diminishing). **Current matmul-free best = exp530 = 1.4731 @ 108.3M** (qk nap4/tph256, v_lut nap6/tph320 d_v16, out nap6/tph1024, residual nap6/tph256).

## Hard v-branch HURTS (v-branch only helps when soft)
Adding the exp507/428 shared-anchor v-branch (qk emits 2*d_qk+d_v, last d_v added to v_lut) but from the HARD argmax qk (exp523) = 1.4901, +0.0076 WORSE than exp513. Contrast: exp508's SOFT v-branch helped ~0.004 (magnitude leakage). So the v-branch benefit was magnitude-leakage-specific; a hard one is just noise.

## Caveats
- Gains are largely capacity-bought: 89.4M (exp513) -> 108.3M (exp530) for ~0.009 bpb. Many per-exp deltas (~2-6mb) sit inside an unmeasured ~5mb seed-noise band (never ran multi-seed). The DIRECTIONAL findings (qk wide-shallow, decoders deep, residual capacity-responsive, hard-v-branch-bad, d_v monotonic) are robust; the exact ranking of the 1.471-1.482 cluster is noise-limited.
- Optimizer (whole line): LION on LUT tables (lut_lr=2e-4, betas 0.9/0.95), AdamW on unembedder(wd=0.1)+tok_emb+norms (adam_lr=3e-4). MeanAbsNorm (L1) for pre/post-norm. See [[project_qk_argmax_recovers_soft]] for the exp475/508/511 origin and [[project_magnitude_leakage_softmax_package]].

## Infra findings (2026-05-24)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ~HALVES the reserved footprint of these soft/LUT runs (e.g. 52GB->24.5GB for the same model) — it's allocator fragmentation, not live tensors. Use it as the default launch flag; enables 2 parallel runs / 2x batch on one H100.
- Soft backward on an E->V sparse-scatter LUT costs ~57GB (materializes huge per-table tensors); use `backward_mode='ste'` (6.6GB) for E->V heads.
- `TinyMHLut(backward_mode='soft')` saves only x + int64 indices in forward and RECOMPUTES the [B,tph,2^nap] tensors in backward (custom autograd.Function), so soft mode's forward activations are small (~1.7GB); peak is set by the backward recompute.
