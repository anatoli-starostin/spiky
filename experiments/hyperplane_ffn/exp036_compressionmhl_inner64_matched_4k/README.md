# exp036 — CompressionMHL (inner=64), param-matched, 4k steps

**First experiment in the CompressionMHL series.** Each block's dense 384→1536→384 GELU FFN
is replaced **entirely** by a `CompressionMHL` bottleneck
(`src/spiky/lutorch/compression_mhl.py`):

```
h = ln2(x)
x = x + compressionmhl(h)          # compress -> FastMHL(inner) -> decompress
```

`CompressionMHL(input_dim=384, output_dim=384, inner_dim=64, nap=6, tph=276, n_heads=1,
forward_mode="hard")`:
- **compress** `Linear(384 → 64)` — projects into the small inner space;
- **FastMHL** — a plain hard `FastMultiHeadLut` operating **in the 64-d compressed space**
  (input = output = 64), full soft-surrogate backward (grads reach the tables and flow back
  to the compressed vector `z`);
- **decompress** `Linear(64 → 384)` — projects back out; its weight is **zero-init'd**
  (residual-identity start).

No parallel dense linear, no GELU, no dense FFN. Attention, LayerNorms, residual, data,
training loop, eval and all hyperparameters are byte-identical to exp032 (d384 / 6L / 6H /
seq 512, device_bs 48, total_bs 24576, n_steps 4096, lr 3e-4, wd 0.1, warmup 0.1,
eval_every 200, seed 1, fp32, vocab 32768, same data).

Idea: keep the (sparse-gradient, param-heavy) table lookup in a **small** inner space so most
of the budget goes to tables while the addressed vector stays low-dimensional.

## Param-match (inner=64, NAP=6 → tph=276)
Per-layer, to match the removed FFN's 1,179,648:
- fixed = compress (384·64+64 = 24,640) + decompress (64·384+384 = 24,960) = **49,600**
- FastMHL budget = 1,179,648 − 49,600 = 1,130,048 → `tph = 1,130,048 / (2^6 · 64) = 275.9`
  → **tph = 276** → FastMHL = 276·64·64 = **1,130,496**
- per-layer total = 49,600 + 1,130,496 = **1,180,096**  (≈ FFN 1,179,648, +448/layer)

| | params |
|---|---|
| exp032 (vanilla, incl. 6× FFN 7,077,888) | 35,792,640 |
| exp036 (CompressionMHL inner=64, tph=276) | **35,795,328** |
| Δ vs exp032 | **+2,688 (+0.0075%)** |

(Confirmed by a build smoke; module unit-tested in `tests/test_compression_mhl.py`.)

## Optimizer
Identical to exp032 except FastMHL LUT-table weights → **no-weight-decay** group (project
lesson). compress/decompress **weights → wd 0.1**, **biases → no-wd**.

## Context — the FFN-slot sweep so far (all 4096 steps)
| exp | FFN slot | params | val_bpb |
|-----|----------|--------|---------|
| exp032 | dense GELU FFN | 35,792,640 | 1.39371 |
| exp033 | Linear ∥ hard-FastMHL | 35,794,944 | 1.41430 |
| exp035 | hard-FastMHL only | 35,792,640 | 1.47106 |
| exp034 | Linear-only | 29,601,792 | 1.47970 |
| **exp036** | **CompressionMHL inner=64** | **35,795,328** | *(this run)* |

## Status
Launched under the owner's GO. Outputs alongside: `metrics.csv`, `summary.json`, `loss.png`,
`checkpoint.pt` (checkpoint gitignored).
