# exp037 — CompressionMHL (inner=64) **+ inner residual**, param-matched, 4k steps

**exp036 with one change: an inner residual inside the CompressionMultiHeadLUT.** Identical
to `exp036_compressionmhl_inner64_matched_4k` in every way except the bottleneck now adds the
compressed input `z` back to the LUT output before decompress:

```
z = compress(x)                 # [N, 64]
y = lut(z).sum(dim=1)           # [N, 64]
y = y + z                       # <-- inner residual (exp037): LUT learns a residual over z
out = decompress(y)             # [N, 384]
```

(exp036 was `out = decompress(lut(z))` with no inner skip.) Enabled via
`CompressionMultiHeadLUT(..., inner_residual=True)`.

The block still uses the standard outer residual `x = x + compressionmhl(ln2(x))`; the inner
residual is *inside* the bottleneck, over the 64-d compressed vector.

## Why
In exp036 the LUT had to reproduce the whole compressed representation from scratch through
the tables. With the inner skip, the LUT only needs to learn a *correction* to `z` — often an
easier target and a cleaner gradient path (z reaches decompress through both the skip and the
lut). This is the standard residual-learning argument, applied inside the compressed space.

## Params — identical to exp036 (inner skip is parameter-free)
The `+ z` add introduces **zero** parameters, so:

| | params |
|---|---|
| exp036 (CompressionMHL inner=64) | 35,795,328 |
| **exp037 (+ inner residual)** | **35,795,328** |
| Δ vs exp036 | **0** |
| Δ vs exp032 (35,792,640) | +2,688 (+0.0075%) |

A clean A/B: same params, same everything, only the inner residual differs. (Confirmed by a
build smoke; module unit-tested incl. `inner_residual=True` in `tests/test_compression_mhl.py`.)

## Everything else — byte-identical to exp036
inner_dim=64, NAP=6, tph=276, hard FastMHL, near-zero table init, per-layer seed 1000+idx,
decompress weight zero-init, LUT tables no-wd / compress+decompress weight wd + bias no-wd;
MinimalGPT+RoPE d384/6L/6H/seq512, device_bs 48, total_bs 24576, n_steps 4096, lr 3e-4,
wd 0.1, warmup 0.1, eval_every 200, seed 1, fp32, vocab 32768, same data.

## Context — the sweep (all 4096 steps)
| exp | FFN slot | params | val_bpb |
|-----|----------|--------|---------|
| exp032 | dense GELU FFN | 35,792,640 | 1.39371 |
| exp036 | CompressionMHL inner=64 | 35,795,328 | 1.40699 |
| **exp037** | **CompressionMHL inner=64 + inner residual** | **35,795,328** | *(this run)* |
| exp033 | Linear ∥ hard-FastMHL | 35,794,944 | 1.41430 |
| exp035 | hard-FastMHL only | 35,792,640 | 1.47106 |
| exp034 | Linear-only | 29,601,792 | 1.47970 |

## Status
Launched under the owner's GO. Outputs alongside: `metrics.csv`, `summary.json`, `loss.png`,
`checkpoint.pt` (checkpoint gitignored).
