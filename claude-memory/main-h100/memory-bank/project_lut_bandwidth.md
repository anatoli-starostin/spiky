---
name: LUT transformer evaluation metric — virtual bandwidth
description: Why LUT transformers are compared on virtual bandwidth, not GPU params or wall-clock
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
LUT operations are inherently sparse: each forward pass only reads one entry per table. On GPU this sparsity gives no benefit (GPUs are optimized for dense matmuls), so GPU wall-clock is not a fair comparison vs dense transformers.

The real metric is **virtual bandwidth** — bits read from memory per inference, accounting for sparsity. On specialized hardware (custom ASICs, neuromorphic chips) where sparse lookups are native, LUTs can be far more efficient than dense matmuls.

**Why:** The research goal is to show LUT transformers can match or beat dense transformers at the same *virtual bandwidth budget*, not the same parameter count.

## Per-primitive virtual bandwidth per forward

For a single LUT module with `n_heads·tph` tables, `table_dim = 2^input_nap` entries per table, `output_nap` votes per entry, per input sample:
- **Bits read** = `n_heads · tph · output_nap · K` where K is the bit-width per weight.

| module | K (bits/weight) | per-sample bits read |
|--------|-----------------|----------------------|
| BitPermutationLUT | 1 | `n_heads·tph·output_nap` |
| MultiBitPermutationLUT K=4 | 4 | `n_heads·tph·output_nap × 4` |
| PermutationalLut fp32 | 32 | `n_heads·tph·output_nap × 32` |
| dense matmul (reference) | 16 (bf16) | `d_in·d_out × 16` |

Example for exp338 out_proj (n_heads=1, tph=2048, output_nap=10) at K=1:
- LUT read: 1·2048·10 = **20.5 kbits per sample**
- Equivalent dense matmul (64→496 via Borda-dominance 32-dim path): d_in·d_out·16 ≈ 64·32·16 = 32 kbits (comparable)

Note: LUT storage is `2^input_nap ×` bigger than the per-sample read — the whole table lives in memory but each sample reads one entry. On a chip with cheap random-access reads, this asymmetry is fine.

## How to apply

- When reporting results, cite both val loss and virtual bandwidth (bits/sample read). A 134M-logical-sign model at K=4 reads **16.8M bits** per sample through out_proj, vs 2.1M at K=1 with same shape — 8× bandwidth overhead for K=4.
- Prefer lower K when it preserves accuracy: K=1 BitPermLUT is 32× cheaper than fp32 PermLut at matched shape.
- When K>1 is necessary (e.g. for out_proj at small tph), prefer MultiBit over fp32 PermLut — same mathematical content but 8× smaller bandwidth.
- Don't equate 134M LUT "logical signs" with 134M dense params. The relevant axis is memory accesses per forward × bit-width per access.
