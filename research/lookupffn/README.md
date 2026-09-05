# LookupFFN — reference study

A clean, heavily-commented, **pure-PyTorch** reference implementation of the LookupFFN
paper, written to be *read and understood*, not to be fast. It deliberately avoids the
paper's compiled C++/CUDA kernels; everything here is CPU-runnable.

- **Paper:** "LookupFFN: Making Transformers Compute-lite for CPU Inference",
  Zeng, Davies, Pulijala, Sankaralingam, Singh. ICML 2024. arXiv:2403.07221.
- **Reference repo (compiled kernels, not used here):** github.com/mlpen/LookupFFN
- **This folder:** `lookup_ffn.py` (the implementation + smoke test) and this README.

## What the paper does (one paragraph)

CPUs have little compute but lots of cache-resident memory, so a dense GEMM feed-forward
network (FFN) is the CPU bottleneck. LookupFFN reformulates the FFN so its work becomes
**memory lookups instead of multiply-accumulate**. A standard FFN computes
`y = sum_i sigma(<x, W_i>) V_i` (score every hidden unit by a dot product, O(d^2)).
LookupFFN instead keeps `K` learnable **hash tables** `T_k` and learnable **hash
functions** `f_k`, and computes `y = sum_k T_k[f_k(x)]`: hash `x` to an integer address
and *look up* one stored row per table, summing over the `K` tables. The hash is a cheap
structured projection (**BH4**, O(d log d)) followed by a "nearest binary code" address;
it is trained with a **softmax relaxation** so the whole thing is differentiable, and it
hardens to a plain gather at inference. Result: ~6–7× fewer FFN FLOPs at near-equal
quality, ~2.5× CPU speedup, and no rehashing (unlike classic LSH such as SLIDE/Mongoose).

## Key equations (as implemented here)

| Eq | Meaning | Where in `lookup_ffn.py` |
|----|---------|--------------------------|
| 6  | `y = sum_k T_k[f_k(x)]` — multi-table lookup replaces the FFN | `LookupFFN`, `LookupTableHead` (hard path) |
| 10 | softmax relaxation over codes → differentiable hash for training | `StructuredHash.soft_weights(top_n=None)` |
| 13 | keep only the top-N codes (efficiency; N→1 ≈ hard) | `StructuredHash.soft_weights(top_n=k)` |
| ~19| `R = prod_i B_i H` — block-diagonal × Hadamard, O(d log d) projection | `BH4`, `fwht` |

Special case worth remembering: if the code matrix `S` is the **full hypercube**
`{-1,+1}^n`, then `argmax_i <z, S_i> = sign(z)`, so the hard address is just the bit
pattern of `sign(z)` — i.e. classic hyperplane locality-sensitive hashing. A smaller
"structured" `S` is just a chosen sub-codebook to shrink the table.

## Mapping: their components → our lutorch analogs

| LookupFFN (paper)                              | Our nanochat-hier / lutorch analog | Note |
|------------------------------------------------|------------------------------------|------|
| Hash projection `z = x R` via **BH4** (O(d log d), structured) | Input **reprojection** matmul into a small subspace (CompressionMHL) | Ours is a small *dense* matmul; BH4 is a cheaper structured/fast-transform projection — a candidate to cut our reprojection FLOPs. |
| Address = nearest binary code `argmax<z,S>` (= `sign(z)` for full `S`) | **HyperplaneMHL** hyperplane sign-tests / anchor-pair sign comparisons, bit-concatenated to an index | Same sign-LSH idea; ours packs sign bits directly, theirs picks nearest of a code set. |
| Differentiable via **softmax over codes** (Eq 10) + top-N (Eq 13); hard gather at inference | Our **hard forward / soft (temperature-scaled) backward** surrogate gradient; `hybrid_smooth` vs `hard` modes | Theirs needs no straight-through/surrogate — a cleaner training story worth trying. |
| `y = sum_k T_k[f_k(x)]`, K tables (Eq 6)       | **FastMultiHeadLut** multi-head gather-and-sum | Same structure: K parallel tables, gather one row each, sum. |
| Table `T_k` stores **full-width** value rows   | LUT weight bank + **output decompression** (CompressionMHL) | Ours decompresses a narrow gathered vector back to full width to save params/bandwidth; theirs stores full rows. |
| FFN only; attention stays dense                | We also hash attention **Value** projections (dual-stream vs uni-stream) | Our scope is broader than FFN-only. |

## How to run the smoke test

Needs a CPU PyTorch (no GPU required, no build step). On nucstar:

```bash
~/projects/spiky-testenv/bin/python research/lookupffn/lookup_ffn.py
```

(or any Python env with `torch` installed). It:
- runs `LookupFFN` forward in **soft** (training) and **hard** (eval/gather) modes,
- checks the soft path is differentiable (BH4 receives gradient),
- checks the fast Hadamard transform is orthonormal,
- runs a tiny 2-layer causal GPT end-to-end,
- prints a per-token FLOP/param comparison vs a dense `d→4d→d` FFN.

At the toy width `d_model=128` the lookup FFN already uses ~3× fewer MACs (for ~1.2× the
params); the advantage grows with `d_model`, because the dense FFN is O(d^2) while the
lookup cost is dominated by `K·(2^n_bits·n_bits + d·block)`, which grows far more slowly.
