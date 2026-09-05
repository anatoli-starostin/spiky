# LookupFFN — reference study

A clean, heavily-commented, **pure-PyTorch** reference implementation of the LookupFFN
paper, written to be *read and understood*, not to be fast. It deliberately avoids the
paper's compiled C++/CUDA kernels; everything here is CPU-runnable.

- **Paper:** "LookupFFN: Making Transformers Compute-lite for CPU Inference",
  Zeng, Davies, Pulijala, Sankaralingam, Singh. ICML 2024. arXiv:2403.07221.
- **Reference repo (compiled kernels, not used here):** github.com/mlpen/LookupFFN
  (mechanism verified against `src/roberta/models/prenorm_lookup/lookup.py` and
  `compute_code_score/kernel.py`).
- **This folder:** `lookup_ffn.py` (implementation + smoke test) and this README.

## What the paper does (one paragraph)

CPUs have little compute but lots of cache-resident memory, so a dense GEMM feed-forward
network (FFN) is the CPU bottleneck. LookupFFN reformulates the FFN so its work becomes
**memory lookups instead of multiply-accumulate**. A standard FFN computes
`y = sum_i sigma(<x, W_i>) V_i` (score every hidden unit by a dot product, O(d^2)).
LookupFFN instead keeps `K` learnable **hash tables** `T_k` and learnable **hash
functions** `f_k`, and computes `y = sum_k score_k(x) * T_k[f_k(x)]`: hash `x` to an
integer address, *look up* one stored row per table, scale it by a scalar, and sum over
the `K` tables. The hash is a cheap structured projection (**BH4**, O(d log d)); the
address is the hard sign pattern of the projection, and a smooth score makes the whole
thing differentiable. Result: ~6–7× fewer FFN FLOPs at near-equal quality, ~2.5× CPU
speedup, no rehashing (unlike classic LSH such as SLIDE/Mongoose).

## The important subtlety: there is NO soft-vs-hard train/inference split

A tempting-but-wrong reading is "softmax-blend over all codes during training, hard gather
at inference" — that would be a train/eval mismatch. **The paper does not do that.**
Training and inference run the **same** forward:

1. **project** `z = BH4(x)` (Eq ~19).
2. **address** `code = bin2dec(sign(z))` — the HARD sign pattern, identical in train and
   eval. With the full-hypercube codebook, "nearest code" == `sign(z)` (classic hyperplane
   LSH), so no explicit code matrix is needed.
3. **score** `m = |z|; score = m.sum() / prod_j(1 + exp(-2 m_j))` — a smooth scalar. This
   is the single dominant softmax-over-codes term, i.e. Eq 10's full softmax collapsed to
   its top-1 term (**Eq 13 with N = 1**); it equals `exp(<z, sign z>) / prod_j (e^{z_j}+e^{-z_j})`.
4. **output** `y = sum_k score_k * T_k[code_k]`.

Gradients flow to the BH4 projection **through the continuous score**, while the discrete
address `sign(z)` is left hard and is the same in both modes. Consequences, all verified
in the official code:
- **train == eval** by construction (same output for the same input; the smoke test checks this).
- **No straight-through estimator** on the address.
- **No temperature** parameter (not fixed, learnable, or annealed).
- **No auxiliary loss** (no entropy / load-balancing / importance / sparsity term); the
  layer returns only its output tensor.

## Key equations (as implemented here)

| Eq | Meaning | Where in `lookup_ffn.py` |
|----|---------|--------------------------|
| 6  | `y = sum_k score_k * T_k[f_k(x)]` — multi-table lookup replaces the FFN | `LookupFFN`, `LookupTableHead` |
| 10 | softmax over codes — the general derivation of the differentiable hash | `lookup_score` docstring (we use its top-1 term) |
| 13 | top-N truncation; **deployed with N = 1** (one code per table) | `lookup_address` + `lookup_score` |
| ~19| `R = prod_i B_i H` — block-diagonal × Hadamard, O(d log d) projection | `BH4`, `fwht` |

## Mapping: their components → our lutorch analogs

| LookupFFN (paper)                              | Our nanochat-hier / lutorch analog | Note |
|------------------------------------------------|------------------------------------|------|
| Hash projection `z = x R` via **BH4** (O(d log d), structured) | Input **reprojection** matmul into a small subspace (CompressionMHL) | Ours is a small *dense* matmul; BH4 is a cheaper structured/fast-transform projection — a candidate to cut our reprojection FLOPs. |
| Address = `sign(z)` bit-packed (nearest full-hypercube code) | **HyperplaneMHL** hyperplane sign-tests / anchor-pair sign comparisons, bit-concatenated to an index | Essentially the same sign-LSH address. |
| **One differentiable forward**: hard `sign(z)` address + smooth magnitude *score* (Eq 13, N=1); train==eval | Our **hard forward / soft (temperature-scaled) backward** surrogate; `hybrid_smooth` vs `hard` modes | This is the key contrast: they get differentiability from a smooth *score* multiplying the row (no STE, no temperature, one forward), rather than a separately-shaped soft backward. |
| `y = sum_k score_k * T_k[f_k(x)]`, K tables (Eq 6/13) | **FastMultiHeadLut** multi-head gather-and-sum | Same structure: K parallel tables, gather one row each, sum. |
| Table `T_k` stores **full-width** value rows   | LUT weight bank + **output decompression** (CompressionMHL) | Ours decompresses a narrow gathered vector back to full width to save params/bandwidth; theirs stores full rows. |
| FFN only; attention stays dense                | We also hash attention **Value** projections (dual-stream vs uni-stream) | Our scope is broader than FFN-only. |
| No temperature, no aux/load-balancing loss     | (n/a)                              | Verified in their code; bucket balance is handled implicitly by the learnable projection + smooth score, not a penalty. |

## How to run the smoke test

Needs a CPU PyTorch (no GPU, no build step). On nucstar:

```bash
~/projects/spiky-testenv/bin/python research/lookupffn/lookup_ffn.py
```

(or any Python env with `torch`). It:
- runs `LookupFFN` and confirms **train() and eval() produce the identical forward** (allclose),
- confirms the BH4 projection still receives gradient (through the smooth score, no STE),
- checks the fast Hadamard transform is orthonormal,
- runs a tiny 2-layer causal GPT end-to-end,
- prints a per-token FLOP/param comparison vs a dense `d→4d→d` FFN.

At the toy width `d_model=128` the lookup FFN already uses far fewer MACs (for ~1.2× the
params); the advantage grows with `d_model`, because the dense FFN is O(d^2) while the
lookup cost is dominated by `K·(4·d·block + code_length)`, which grows far more slowly.
