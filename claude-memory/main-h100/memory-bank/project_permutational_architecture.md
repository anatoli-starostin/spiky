---
name: Permutational LUT Architecture
description: Ranking-based (no-residual) LUT transformer — original design (exp247-249) and its evolution into the FullBitPermRankAttn_ctx128 family (exp299+)
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
# Permutational LUT Architecture

## Core insight (still relevant)

Residual connections break permutational consistency: `rank(x + y) ≠ f(rank(x), rank(y))`. LUTs in non-smooth mode operate purely on rankings (anchor-pair comparisons `sign(x[a] − x[b])`), so the residual stream's additive accumulation is fundamentally incompatible with the downstream rank-sensitive operations.

**How to apply**: designs should avoid addition in the main data path. Positional embeddings added before Q/K are an acceptable exception (they enter once per layer, not in an iterative path).

## Original architecture (exp247–249)

Per layer: Q/K LUT(nap=5, tph=128) → QK-LayerNorm → SDPA → OutProj LUT(nap=5, tph=256) → LayerNorm → next layer. No FFN, no residuals, 5.56M params. Backward ranking-prediction loss (pairwise BCE over 496 pairs for E=32) kept gradients flowing through 6 layers without residuals. First successful pure-ranking 6-layer training — CE ~1.88 at step 6k.

## Evolution into FullBitPermRankAttn_ctx128 (exp299+)

The exp247-249 architecture matured into the modern family used in exp299–exp350+:

1. **`BitPermutationLUT`** replaced the original MultiHeadLut-based LUT primitive. 1-bit packed weights, fused CUDA forward/backward, fp8/bf16 latent + STE gate. Produces pair-dominance output `[B, H, P]`.
2. **`DominanceCanonicalize`** on Q/K (instead of raw QK-LN): Borda projection + LN + rank-projection back to ±1. Cleans up bit-vote outputs before SDPA.
3. **`DominanceToVector`** (Borda + LN in one module) after attention (d_v) and after out_proj (E). Replaces manual einsums + separate LayerNorms.
4. **`CANONICAL_FULL_COVERAGE`** anchor-sampling (guaranteed pair coverage via tile-and-repair) — default.
5. **`partition_sets`** for out_proj: restricts the `H·d_v`-wide input's anchor pairs to within-head (no cross-head ordering — different heads have no comparable scale).
6. **No explicit ranking-prediction loss** needed: the modern pipeline's gradient flow through STE-gated BitPermLUT + DominanceCanonicalize + SDPA is sufficient for 6 layers without the auxiliary loss.

## Key findings about the rank-flow architecture

- **Scale carries signal too**: the original assumption "only ordering matters" is not quite right. SDPA sharpness varies per-head per-sample, which shows up in the magnitudes downstream modules see. For out_proj this is why **partition_sets helps** — cross-head ordering has no stable semantic meaning; within-head ordering does.
- **Q/K want crisp signs, out_proj wants graded weights**: the modern hybrid (exp347) uses 1-bit q/k/v and 4-bit MultiBitPermutationLUT out_proj. All-K=4 (exp349, exp350) regresses.
- **Layer 0 needs more capacity**: SDPA at layer 0 is heavily local (78% attention within 4 tokens), so its out_proj sees information-dense input. Per-layer difficulty ranges 0.887→0.986 sign-acc at fixed distill budget. Per-layer graded capacity is a natural next step.

## Status

- **exp329** (all-BitPermLUT + CFC, 210M bits): val 1.379 @ 25k. Current accuracy target.
- **exp347** (hybrid: BitPermLUT q/k/v + MultiBit K=4 out_proj, tph=256): val 1.432 @ 25k. Best tph-efficient setup.
- Distill framework `transformer_exps/distill_exp338/` gives a clean way to sweep out_proj shapes without full 25k-step retraining.

See [project_transformer_exp_summary](project_transformer_exp_summary.md) for full numbers.
