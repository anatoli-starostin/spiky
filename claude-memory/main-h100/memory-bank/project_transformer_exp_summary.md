---
name: LUT Transformer Experiment Summary
description: Key results from the BitPermutationLUT / PermutationalLut / MultiBitPermutationLUT era (exp299–exp350+) and the distill_exp338 methodology
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
# LUT Transformer Experiments (exp299 onwards)

Architecture family: **FullBitPermRankAttn_ctx128** (6 layers, E=32, 4 heads, d_qk=24, d_v=16, ctx=128, vocab=257).

## End-to-end Results (val loss, 25k steps unless noted)

| exp | out_proj variant | val | notes |
|-----|------------------|-----|-------|
| exp329 | BitPermLUT (in=10, tph=2048, on=10), no partition | **1.379** | CFC default, reference |
| **exp338** | BitPermLUT (in=10, tph=2048, on=10), +partition_sets | 1.401 | Partition alone slightly worse |
| exp332 | BitPermLUT + partition_sets (narrower) | 1.412 | |
| exp336 | BitPermLUT (in=10, tph=256, on=16) | 1.494 | Small tph hurts BitPermLUT |
| exp342 | BitPermLUT (in=4, tph=8192, on=10) | 1.529 | Distill-winner shape — but LM task regresses |
| exp343 borda | PermLut fp32 (in=10, tph=256, on=32) + borda scale | 1.450 | Smooth-rational closes gap to exp338 by 30% |
| exp346 | PermLut + vote_quant_levels=16 (4-bit after rational) | 1.470 | K=4 quant costs only 0.02 val-loss |
| **exp347** | **MultiBit K=4 (midrise) out_proj only** | **1.432** | Hybrid: q/k/v 1-bit, out_proj 4-bit. Matches PermLut_q16 territory. |
| exp349 | All-MultiBit K=4, qk_tph=192 | ~1.48 | All 4-bit underperforms — q/k "too heavy" |
| exp350 | All-MultiBit K=4, qk_tph=64 | 1.492 | Reduced q/k tph partially helps but still worse than exp347 |

**Takeaway**: BitPermLUT (1-bit) with many tables still wins end-to-end at this scale. MultiBit K=4 on out_proj alone (hybrid) gets close. Uniform K=4 across q/k/v hurts — q/k don't need wider weight precision.

## Distillation Framework (`transformer_exps/distill_exp338/`)

**Methodology**: extract `(out_proj_input, out_proj_output)` pairs from exp338's trained model (102K samples/layer via `collect.py`), train candidate LUT architectures on layer N to match the **pair-wise sign of the Borda-projected output** (32-dim). Rank-based loss (`pair_soft_sign` + MSE) — matches what the next layer's TinyAnchorPairsLookup actually reads.

Key lessons during development:
1. **Target must be rank of Borda-projected E-dim output**, not raw 496-dim pair dominance. Matching the latter overconstrained (464-dim null space); CUDA forward output needed to be divided by √(E−1) to stabilize.
2. **Non-determinism fixed** — see `src/spiky/lutorch/determinism.py`. Seeded data sampling in `TextSnippetSampler`, plus dispatching `tiny_apl_bwd` and `bit_perm_lut_weight_grad` via PyTorch's sort-based scatter under `set_deterministic(True)`. Yields bitwise-identical checkpoints across launches (with `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`). ~6.5% per-step slowdown.

### Per-layer difficulty (at fixed distill budget 1.3M bit-params, candidate c_a_in04_tph8192_on10)

| layer | best sign_acc |
|-------|---------------|
| 0 | 0.887 |
| 1 | 0.937 |
| 2 | 0.949 |
| 3 | 0.977 |
| 4 | 0.982 |
| 5 | 0.986 |

**Mechanism**: layer 0's SDPA attention is heavily local (78% mass within 4 tokens vs 19% for layer 3 — see `attention_stats.png` in exp338 folder). Its out_proj input is information-dense per-head. Later layers process already-smoothed features.

### Layer-0 Pareto (BitPermLUT)

| bit-params | best | arch |
|-------|-------|------|
| 21M (+teacher pairs) | 0.9903 | teacher-shape ceiling |
| 21M (random pairs) | 0.985 | teacher-shape |
| 10.5M | **0.972** | in=7, tph=8192, on=10 |
| 8.4M | 0.962 | in=6, tph=4096, on=32 |
| 2.1M | 0.908 | in=5, tph=2048, on=32 |

### Layer-0 Pareto (PermLut)

| logical signs | best | arch |
|-------|-------|------|
| 21M | **0.9991** | teacher-shape (in=10, tph=2048, on=10) |
| 10.5M | 0.9981 | in=7, tph=8192, on=10 |
| **8.4M** | **0.9956** | **in=10, tph=256, on=32 (PermLut p20)** |
| 5.2M | 0.9917 | in=10, tph=512, on=10 |

### MultiBit vs PermLut (matched shape in=10, tph=256, on=32)

| module | best sign_acc | storage |
|--------|---------------|---------|
| PermLut fp32 | 0.9956 | 8.4M × 32 bits = 268Mbits |
| PermLut + vote_quant_levels=16 | ~0.991 | fp32 weights, 4-bit votes |
| **MultiBit K=4 (midrise + rational T=0.1)** | **0.9898** | 8.4M × 4 bits = 34Mbits (8× smaller) |

## Module Primer

### BitPermutationLUT (1-bit)
- `src/spiky/lutorch/bit_permutation_lut.py`. Emits pair dominance [B, H, P]. Needs `DominanceToVector(E)` (Borda+LN) downstream.
- Trains via `BitPermutationLUTOptimizer` (STE gate `T/(T+|latent|)²` + fp8/bf16 Adam).

### PermutationalLut (fp32)
- `src/spiky/lutorch/permutational_lut.py`. Soft-rational forward `0.5·x/(T+|x|)` → scatter_add (or sparse matmul). Emits E-dim vector directly (no Borda).
- Standard `torch.optim.Adam` on `nn.Parameter` weights.
- `borda_scale_mode='borda'` (default) divides scatter by √(E−1) to match BitPermLUT's post-LN scale. Under old `'clt'` scale, pre-LN magnitudes were ~4× smaller.
- `vote_quant_levels=N`: STE quantize rational output to N levels before scatter.

### MultiBitPermutationLUT (K∈{2,4,8})
- `src/spiky/lutorch/multi_bit_permutation_lut.py`. Same output format as BitPermLUT (pair dominance). Storage: K-bit signed ints packed into int32 blocks (`32/K` slots per int32).
- **Rational-pre-quant** (T=0.1 default): pack applies `rational(latent)` before quantising — matches PermLut+q_N semantics for K=log₂(N).
- **Midrise quantiser**: no level at 0 (levels at (2q+1)/(2·2^(K−1))). Fixes the "dead zone at 0" that slows bootstrap for small init.
- Per-pair bias added in forward to compensate for midrise shift (precomputed from inv_idx).
- Trains via `MultiBitPermutationLUTOptimizer` (bf16 latent + STE + `multi_bit_pack` repack per step).

### CANONICAL_FULL_COVERAGE policy
Default input-anchor sampling. Guarantees each canonical pair is covered at least once when slots ≥ P = C(N,2). Uses tile-and-repair algorithm (concatenated randperms + greedy swap for intra-table duplicates).

### partition_sets
Restricts anchor pairs to within-partition. Used in out_proj where input is `H·d_v` concatenated heads — shrinks pool from C(64,2)=2016 to 4·C(16,2)=480 for our config.

## Shape Lessons

- **Narrow × many** (BitPermLUT, 1-bit): wins at fixed budget. `in=4, tph=8192` best at 1.3M bits.
- **Fat × few** (PermLut / MultiBit, continuous): wins for quantized/continuous votes. `in=10, tph=256, on=32` best at 8M logical signs.
- **Multi-LUT stacks (2L, 3L)**: consistently lose vs single-LUT at matched budget in distillation.
- **Output_nap**: diminishing returns above ~12 for BitPermLUT; PermLut handles on=32+ well.
- **Per-layer graded budget**: should help — layer 0 needs 8× more capacity than layer 5. Not yet tested in e2e.

## Open Problems

- **MultiBit still needs ~2× warmup vs BitPermLUT**. Midrise + init tuning helped but didn't eliminate.
- **All-MultiBit underperforms hybrid**: q/k don't benefit from K>1 — they feed SDPA's dominance-cleaning step which wants crisp signs.
- **Per-layer graded budget** untested in e2e.
