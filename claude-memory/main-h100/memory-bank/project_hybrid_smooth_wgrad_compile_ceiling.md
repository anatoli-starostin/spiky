---
name: hybrid-smooth-wgrad-compile-ceiling
description: "FastMHL hybrid_smooth weight grad is at torch.compile's ceiling ~17 ms at LUTGPT shapes; custom triton kernel attempted in another session and failed to beat it. Don't retry."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# `_hybrid_smooth_weight_grad` is at the torch.compile fusion ceiling

At the LUTGPT bench shape (B=12288, n_tables=512, K=128, n_outputs=384) the function
`_hybrid_smooth_weight_grad` in `src/spiky/lutorch/fast_multi_head_lut.py` clocks
**17.0 ms** — and that's the realistic floor.

## What compile is already doing

Micro-bench evidence (2026-06-06):
- Caching the `arange(n_tables) * K` offset as a buffer: 17.00 → 17.00 ms (compile constant-folds it).
- Precomputing `main_flat = main_index + offset` outside the function: 17.00 → 17.02 ms (compile fuses the offset add into the scatter's address computation).
- Precomputing the source `(1-u) * grad_pt` outside the function: 17 → **51.6 ms** (3× slower).

The third bullet is the key insight: **compile is fusing `(1-u) * grad_pt + index_add_` into a single streaming kernel** that reads grad_pt once, multiplies, and atomic-adds without ever materialising the [B, n_tables, n_outputs] = 9.66 GB intermediate. The eager version materialises 9.66 GB + reads it again = ~20 GB needless round-trip × 2 scatters = ~35 ms penalty.

Peak alloc inside the function is only `grad_pt + 0.28 GB` of buffers — no temp materialisation.

## Alternatives tried, all worse or unusable

| Approach | Time | Peak alloc | Notes |
|---|---|---|---|
| Current (compile-fused 2× atomic) | 17.0 ms | 10.0 GB | baseline |
| einsum fp32 (build [B, n_tables, K] selection mass + bmm) | 15.9 ms | 16.5 GB | +6.5 GB memory tax kills it |
| einsum bf16 | 9.3 ms | 16.7 GB | broken: max_diff 0.17 from bf16 input precision over ~192 contributions/cell |
| bucket-partial (4 buckets) | 22.6 ms | 11.2 GB | slower; kernel launch overhead dominates |
| F.embedding_bag(per_sample_weights) in fwd | +6 ms | −1 GB | regression: dedicated kernel slower than compile's pointwise fusion |
| Drop `g32 = grad_pt.float()` cast | 17.0 ms | 10.0 GB | no-op: grad_pt is already fp32 |
| Mixed bf16 source → fp32 dest in `index_add_` | — | — | not supported by PyTorch kernel |

## Custom triton kernel: previously tried, failed

**A custom triton kernel for the dual-scatter was attempted in a prior session and failed to beat torch.compile.** Don't burn cycles trying again. The compile fusion is genuinely competitive with handwritten CUDA at this shape.

## Where the 17 ms goes

- ~5–6 ms HBM-bandwidth floor: 2 × `grad_pt` (9.66 GB) reads. Can't avoid both because each scatter needs its own pass with different weights.
- ~11–12 ms atomic-add contention: 2 × 2.4 B atomics over a 25 M-cell `grad_w_flat` destination.

Either could be reduced in principle, but the triton attempt already showed the practical floor is right where compile puts it.

## How to apply

- Treat 17 ms as the load-bearing cost of `_hybrid_smooth_weight_grad` at LUTGPT shape; don't propose further PyTorch-level rewrites of it.
- For G mode (FastMHL hybrid_smooth + ball): total ~30 ms is roughly 17 ms wgrad + 10 ms ball-bwd + 3 ms hybrid_smooth fwd. Each component is already optimised.
- If a deployment-quality fast LUT is needed, use H mode (`forward_mode='hard', backward_mode='ball'`) per [[fastmhl-hard-ball-deployment-sota]] — natively trained-hard, no weight-grad doubling, lands at 21 ms.

Cross-refs: [[fastmhl-hard-ball-deployment-sota]].
