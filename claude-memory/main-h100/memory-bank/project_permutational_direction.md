---
name: Permutational LUT strategic direction
description: Quantized-weight LUTs (BitPermLUT / MultiBitPermLUT) are the priority — hardware-friendly inference (K-bit packed storage) outweighs small accuracy gaps vs fp32 PermLut
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
The design space has three primitives, ordered by storage cost:
1. **BitPermutationLUT** — 1-bit weights, best per-bit accuracy at same tph, best inference efficiency.
2. **MultiBitPermutationLUT** — K-bit packed (K∈{2,4,8}), rational-pre-quant + midrise quantizer. Mathematical equivalent of `PermutationalLut + vote_quant_levels=2^K` but stored as packed K-bit ints.
3. **PermutationalLut** — fp32 smooth-rational. Best accuracy per "logical sign" but 32× storage overhead.

## Current stance

- **BitPermutationLUT stays primary** for q/k/v where crisp ±-ish dominance feeds SDPA and subsequent DominanceCanonicalize. K>1 here is overkill and empirically slows training (see exp349/350 vs exp347).
- **MultiBit K=4 for out_proj** is the interesting direction — exp347 (hybrid BitPermLUT q/k/v + MultiBit K=4 out_proj, tph=256) reaches val=1.432 vs exp338 BitPermLUT (tph=2048) at val=1.401, with **8× less storage** for out_proj. Closes most of the gap.
- **PermutationalLut is useful for distillation ceiling-testing** and as the "smooth" reference, but not the target deployment primitive.

## Why K-bit quantized (not fp32)

- **Storage**: K-bit packed into int32 blocks — 8× smaller than fp32 for K=4, 32× smaller than fp32 for K=1.
- **Inference**: sum of K-bit signed ints stays in int32, no float arithmetic in the hot path.
- **Training dynamics**: rational-pre-quant at T=0.1 (matches PermLut's smooth backward) + midrise quantizer (no dead zone at 0) gives clean STE training with ~2× warmup vs 1-bit.

## How to apply

- For new architectures, default to BitPermLUT for small-fan-out modules (q/k/v) and MultiBit K=4 for wide-fan-out projection modules (out_proj, FFN).
- Use PermLut as the oracle for distillation experiments (best accuracy per shape) and for sanity-checking whether a shape is learnable at all.
- Don't invest in fp32 PermLut for production; the 0.01-0.02 val-loss gap vs MultiBit K=4 doesn't justify the 8× storage cost.
