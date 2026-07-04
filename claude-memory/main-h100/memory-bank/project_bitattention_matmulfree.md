---
name: Matmul-free LM via LUTs and BitAttention
description: Why exp415 demonstrates that an LM can run end-to-end on bit-LUTs + BitAttention with no float matrix multiplications, and what scales to large vocab
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
## Architectural claim

A language model end-to-end can run with **no float matrix multiplications**, using only:
- Bit-LUT lookups (`BitPermutationLUT` / `MultiBitMultiHeadLUT`) for token embed → Q/K/V → out_proj → unembedder.
- **BitAttention**: SDPA on ±1 dominance vectors reduces to integer popcount + ±1-weighted accumulation.
- Elementwise float ops only: softmax, LayerNorm in `DominanceToVector`, the final `α·logits + bias`.

exp415 (val=1.372 short-scale, 24k×bs=8) is the existence proof at V=257.

## What "BitAttention" replaces

Standard SDPA:
```
attn = softmax(Q @ K^T / sqrt(d)) @ V    # two matmuls over float vectors
```
After `DominanceCanonicalize` on Q and K, both are ±1 vectors of length P_qk = C(d_qk, 2). V is similarly ±1 of length P_v = C(d_v, 2). The two matmuls reduce to:

| Op | Naive | LUT-paradigm equivalent | Why |
|---|---|---|---|
| `Q @ K^T` | float × float matmul | `popcount(Q XOR K)` over P_qk bits | Q, K ∈ {-1, +1}^P_qk → inner product = P_qk − 2·Hamming |
| softmax | float exp/sum | float exp/sum (unchanged) | Genuine non-bit op, but small (per-position) |
| `attn @ V` | float × float matmul | softmax-weighted ±1 accumulator | V ∈ {-1, +1}^P_v → integer/fp accumulation |

For exp415-scale (P_qk=276, P_v=120):
- `Q @ K` is a 276-bit popcount per (i, j) — 5 instructions on `__popcll`.
- `attn @ V` is a 120-element ±1 weighted sum.

On hardware with native popcount + integer accumulation (every modern CPU/GPU), BitAttention executes with 128–512× the operand density of float SDPA.

## Other architecture pieces

- **Q/K/V projections**: `BitPermutationLUT(E_in → d_qk or d_v, n_heads=4)` per layer. Output is pair-dominance.
- **Out projection**: `BitPermutationLUT(H·d_v → e_out, n_heads=1)` per layer.
- **Token embedder**: `nn.Embedding` (table lookup, not matmul).
- **Unembedder (the missing piece, addressed by exp415)**:
  - Pre-exp415: dominance LUT + `Linear(P, V)` (matmul) — exp409.
  - Pre-exp415 alt: dominance LUT + `MultiHeadLut(d, V)` (LUT, but two-stage with Borda+LN in between) — exp410.
  - **exp415: single `MultiBitMultiHeadLUT(192 → 257, K=4, output_nap=64, tph=3072)` directly emits raw vote sums + per-token bias + scalar scale → logits.** No matmul. Val=1.372 (vs exp409's 1.379, exp410's 1.428, exp407 MLP baseline's 1.334).

The architectural gap to a wide-MLP baseline is **0.04 nat** at this scale.

## Forward graph float work (exp415)

The remaining float ops, in order:
1. `nn.Embedding` — gather, no matmul.
2. **Per layer (×6):**
   - `qk_input_ln` (LayerNorm on E_in + pos_emb dim) — vector elementwise.
   - `attn_to_vec` (`DominanceToVector` post-attention): tiny einsum (Borda projection on d_v=16) + LayerNorm.
   - `out_to_vec` (`DominanceToVector` post-out): tiny einsum (Borda on e_out=32) + LayerNorm.
   - softmax(Q@K) — per (B, H, T_q) row, length T_k.
   - softmax-weighted accumulation of V (the second SDPA matmul, on bit V).
3. Final: `logits = lut_out · α + bias` — elementwise.

All of this is O(B·T·d_small) where d_small ∈ {16, 24, 32, 257}, never E·V or E·E.

## Scaling to large vocab (V = 32k)

Direct-vocab LUT unembedder at exp415-quality (vpo=765) requires:
- **Inference**: `bit_weights` ≈ **400 MiB** (4-bit packed, ship-only). Competitive with quantized MLP heads.
- **Training**: bf16 latent + fp32 Adam state ≈ **11.6 GiB**. Heavy but fits on H100.

Path to cheaper training:
- Extend `MultiBitMultiHeadLUT` to fp8 latents (already supported in `BitPermutationLUT`) → 2× compression.
- Lighter optimizer (SignSGD-style on bit_weights, no fp32 m+v) → 4× compression.
- Combined → ~1.5 GiB for V=32k unembedder training.

## Implications

1. **The matmul-free architecture is real, not just rhetorical.** exp415 demonstrates it works; the cost is 0.04 nat at small vocab.
2. **Inference deployment is competitive at any V.** Only `bit_weights` ships; ~400 MiB at V=32k for vpo=765.
3. **Training memory is the engineering challenge** at large V, not architecture. fp8 latents + cheap optimizer is the path forward.
4. **Hardware story**: dedicated bit-LUT silicon (or even modern CPUs with popcount) executes the bulk forward in bit operations, with float work confined to softmax + LayerNorm + small Borda projections. Expected speedup vs float matmul: depends on hardware, but the operand density is 16–32× per-bit advantage.

## Full-scale results (V=257, 100k×bs=32)

| Exp | Architecture | Final val | Per-token weight read |
|---|---|---|---|
| exp243 (vanilla) | dense fp32 transformer (matmul everywhere) | **1.2031** | ~153 Mbit |
| exp370 | LUT transformer + MLP unembedder, **has magnitude leak** in LUTBlock | 1.2035 (artifact) | ~11.6 Mbit |
| **exp390** | LUT transformer + MLP unembedder, **leak-fixed** (out_canon) — fair full-scale baseline | **1.2166** | ~11.6 Mbit |
| exp426 | matmul-free LUT-only (exp415 scaled to 100k×bs=32, leak NOT yet fixed) | 1.2545 | **~1.68 Mbit** |

The leak fix: exp377/exp390 add `DominanceCanonicalize(E)` after `out_proj` in `LUTBlock`, before `out_to_vec`, so that gradient carriers `x[a]−x[b]` flowing back through downstream anchor lookups have ±1-bounded magnitudes (the forward `sign()` is already magnitude-agnostic; the leak is on the backward path). Cost: +0.013 nat at full scale, but the architecture is principled.

Honest gap analysis (matmul-free vs fair baseline at the same compute):
- exp426 (matmul-free, leak still present) vs exp390: **+0.038 nat behind** — combination of "matmul-free unembedder" + "leak still present in LUTBlock".
- Disentangling experiment (TODO): fork exp426 with out_canon restored. If it lands ~1.22 the leak was the bigger contributor; if it stays ~1.25 the matmul-free unembedder is the gap.

## Caveats

### Bandwidth caveat — cache residency at small scale

The naive per-token bandwidth comparison (LUT ~91× cheaper than vanilla matmul) overstates the practical advantage **when weights fit in cache**. At V=257, both designs are cache-resident on modern GPUs:

| Design | Total weight footprint | Fits in cache? |
|---|---|---|
| exp243 vanilla (fp32 params) | 18.6 MiB | ✓ H100 L2 (50 MiB), ✓ A100 L2 (40 MiB) |
| exp390 (bit_weights + continuous) | 27 MiB | ✓ H100 L2, borderline A100 L2, ✓ Intel/AMD L3 |
| exp426 (bit_weights + continuous) | ~26 MiB | same as exp390 |

In the cache-resident regime, the matmul "reads its full weight matrix per token" framing is wrong — weights are read once and reused via batching. Same for LUT lookups: once `bit_weights` is in L2, each per-token table lookup is an L2 read, not an HBM read. The 91× advantage shrinks.

The LUT bandwidth advantage actually materializes when:
1. **Weights don't fit in cache** — large vocab, large d_model, or scaling exp426 to V=32k (latent grows to 1.6 GiB, well past L2/L3).
2. **Per-token inference, no batching** — edge/single-user serving where matmul can't amortize across batches. Bandwidth advantage stays ~50–100× even with cache.
3. **Custom hardware that lacks tensor cores** — bit-LUT-native silicon. Per-token throughput becomes bandwidth-bound, and LUTs win on operand-bit density (1-bit vs bf16 = 16× more operands per byte).

So the realistic LUT design promise is: **inference deployment on bandwidth-bound or non-matmul-friendly hardware**, not GPU training (where matmul has every advantage). At V=257, on H100, the LUT design loses on training time (3.3h vs 0.4h vanilla) and gains nothing on inference latency — the architectural argument has to be made on the bit-storage compression for deployment to constrained devices, or on theoretical hardware that has bit-LUT primitives.

### Other caveats

- Tests so far are at byte-level V=257 — small vocab makes direct-vocab LUTs feasible without heroics.
- The 0.04 nat gap at V=257 short-scale (and 0.038 nat full-scale) may widen at full LLM scale; not yet measured.
- BitAttention as named here is the natural specialization of SDPA to ±1 inputs; not a new operator, just a recognition that the existing F.scaled_dot_product_attention call collapses cleanly when fed dominance bits.

## Related

- See `project_lut_bandwidth.md` for the virtual-bandwidth metric used to compare LUT vs dense compute fairly.
- See `project_transformer_exp_summary.md` for the chain of experiments that led to exp415.
