# inner=48 v_lut sweep — results (task ae659700)

**Headline:** a routed V (CompressionMHL) at cheap width inner=48, in the **NO-DECOMPRESS** regime (inner_out=−1, n_heads=8), **BEATS the dense Linear V** — at both short (3500) and full (16k) length.

Anchor = adjusted-exp_n_0084 DENSE-V, same schedule (short-run 1.3979 @3500). Full-length dense targets: exp_n_0084 1.1987, exp_n_0045 (tied) 1.1977.

## FINALE (full 16k of the winner)
| config | total params | v_lut params | val_bpb @16k | vs dense |
|---|---|---|---|---|
| **no-decompress H8** (in48/inner_out=−1/tph88/nap6/V0.50) | 171,163,032 | 104,696,076 | **1.19223** | **−0.0065 vs 0084 (1.1987); −0.0055 vs 0045 (1.1977) — BEATS BOTH** |

## Short-run sweep (@3500, vs dense anchor 1.3979)
DECOMPRESS regime (inner_out=48, n_heads≤4) — never matches dense:
| axis | best | val_bpb | vs dense |
|---|---|---|---|
| init V-scale | 0.50 (=0.544) | 1.4215 | +0.0236 |
| (0.388 too low +0.0276 · 0.65 +0.0249) | | | |
| tph | 88 | 1.4215 | +0.0236 (44 +0.0271, 176 +0.0322, 352 +0.0257 — flat/noisy) |
| heads | H4 | 1.4215 | +0.0236 (H2 +0.0437 much worse) |

NO-DECOMPRESS regime (inner_out=−1, n_heads≤8):
| config | val_bpb | vs dense |
|---|---|---|
| **D1 H8/tph88** | **1.3958** | **−0.0021 (BEATS dense @3500)** |

## Winner param breakdown (per layer ×6)
compress 147,840 (Linear 384→384) + table 17,301,504 (704×64×384; cells at full output width) + decompress 0 + temps 2 = 17,449,346/layer → 104,696,076 total.

## Quality-vs-FLOP verdict
The no-decompress corner wins by spending **memory bandwidth / params** (a big gathered 104.7M table, ~270K floats/token/layer read) instead of **matmul FLOPs**: its compute is just one small compress matmul (≈ dense V's own matmul) + gather + sum — NO decompress matmul. So at ~equal compute-FLOP to dense (but ~118× the params/layer), routed-V beats the dense Linear V. The cheap decompress regime (small table) never matches dense.

## Recommendation
- The no-decompress H8 config is a genuine win over dense V at equal compute-FLOP; the cost is param/bandwidth (2.5× dense params). Worth pursuing where memory is cheaper than FLOPs.
- Follow-ups: D2 (no-decompress tph176 — does more tables help further, at more params), and inner_in sweep in the no-decompress regime (compress width vs table size tradeoff).

See finale_vs_dense.png (16k curves) and vsweep_summary.png (prior wide-inner sweep).
