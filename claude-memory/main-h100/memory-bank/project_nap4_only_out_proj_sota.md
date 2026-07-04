---
name: nap4-only-out-proj-sota
description: "NAP=4-only out_proj with tph=2048 (param-matched to baseline) achieves new bs=16 SOTA = 1.6009 bpb. Beats single-NAP=8 baseline by 20.6 millibits and multi-NAP variant by 8.8 millibits. Pays 16× LUT bandwidth at inference. Reframes the \"row-collapse\" pathology as a \"gradient-coverage at small batch\" problem."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# NAP=4-only out_proj — new bs=16 SOTA (2026-05-17, exp390)

**Setup**: fork of exp387 (multi-NAP SOTA). Single change: out_proj is a single TinyMHLut(soft) with NAP=4, tph=2048 (vs baseline NAP=8 tph=128). Param count exactly matches baseline (1 head × 2048 tph × 16 rows × 48 outputs = 1,572,864 per layer).

**Result**: **exp390 = 1.6009 bpb @ 43.06 M params**, 8000 steps, bs=16. 

| exp | val bpb | params | Δ vs exp365 | bandwidth | inference |
|-----|---------|--------|---|---|---|
| exp365 (single NAP=8) | 1.6215 | 43.06 M | — | 1× | 1 lookup/table × 128 tables |
| exp387 (multi-NAP) | 1.6097 | 42.47 M | −0.012 | 2.25× | 1 lookup × 288 tables |
| **exp390 (NAP=4 tph=2048)** | **1.6009** | 43.06 M | **−0.0206** | **16×** | 1 lookup × 2048 tables |

## Per-token bandwidth comparison (out_proj only)

| | tables | rows/table | bytes/lookup | bytes/token/layer |
|---|---|---|---|---|
| exp365 | 128 | 256 | 192 | 24,576 |
| exp390 | 2048 | 16 | 192 | 393,216 (**16×**) |

Same total LUT memory (1.57M params/layer). But exp390 reads 16× more entries per token because it queries 16× more tables. **Real bandwidth tradeoff.**

## Why this works — the gradient-coverage hypothesis

User insight that reframes the row-collapse story: it's not about "the model concentrates on few rows" but about **gradient coverage at small batch**.

With NAP=8 at bs=16:
- Each table has 256 rows.
- 8192 tokens/batch / 256 rows = ~32 tokens per row IF distribution were uniform.
- But argmax peaks the distribution → many rows get 0 tokens per batch.
- No gradient → those rows never train → routing collapses around the rows that DID get gradient at init.

With NAP=4 at bs=16:
- Each table has 16 rows.
- 8192 / 16 = ~512 tokens per row uniform.
- Even with peaked distribution, every row gets many tokens.
- Gradient on every row every batch → routing trains properly.

The exp364 (bs=192) result confirms the framing: at bs=192 with NAP=8, every row gets enough coverage to train. The "ideal" few-NAP=8-tables solution exists — but requires large batch to discover.

## Three operating points the user identified

1. **Train with grad_accum at bs=16** (exp367 proved bs=16+accum=8 ≡ bs=128 trajectory). Pays 8× training wall-clock, free inference. Untested at full 8000 steps.
2. **Distill from bs=192 SOTA (exp364 = 1.3769)** into bs=16 NAP=8 student. Teacher's per-token soft outputs bypass row-coverage starvation. Untested.
3. **Pay inference bandwidth (exp390 / exp387)** to skip the coverage requirement. Tested and works, but 2.25× or 16× bandwidth penalty.

## How to apply

- **exp390 is the bs=16 SOTA but with a significant bandwidth cost**. Whether to adopt depends on the deployment context:
  - For research / pure-quality comparisons: use exp390.
  - For hardware-efficient LUT inference: exp365 baseline or exp387 (2.25× bandwidth, less collapse) is more representative of the LUT design intent.
- The interesting follow-up is **distillation from exp364** — has potential to deliver exp390-class accuracy at exp365 bandwidth. exp364 checkpoint is saved at `/home/starost/spiky/nanochat_exps/exp364_bs192/checkpoint.pt`.
- Also worth a **bandwidth-budget sweep**: NAP=4 tph∈{256, 512, 1024} to find the knee in the bandwidth/quality curve. exp390 may be over-budget on bandwidth.

## Why exp388 (multi-NAP everywhere) failed

When applied to qkv (100% touched at L0), v (99%), residual (99% at most layers) — modules that ALREADY had good gradient coverage at bs=16 with their existing NAP — multi-NAP just fragments the parameter budget. There was no coverage problem to fix, so the workaround didn't help. Multi-NAP / high-tph-low-NAP is a targeted fix for the gradient-coverage pathology, not a general regularizer.
