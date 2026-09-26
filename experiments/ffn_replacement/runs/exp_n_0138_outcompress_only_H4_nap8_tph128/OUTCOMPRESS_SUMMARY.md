# Output-compression-only ablation (inner_in=−1) — full 16k (exp_n_0138)

exp_n_0121 with inner_in 48→−1: no input compress (route on the full 384-d input), keep the decompress
Linear (tables emit 48, concat 192 → 384). **final val_bpb 1.21249** (best 1.21229). Isolates the
decompress (output) projection on top of raw 384-d routing.

## The projection ablation ladder (all H4/nap8/tph128, UNTIED, 16k)
| config | input compress | output decompress | FFN params | FFN-FLOP | vBW | val_bpb | Δ vs vanilla (1.20144) |
|---|---|---|---|---|---|---|---|
| **0121 full CompressionMHL** | ✓ (48-d) | ✓ (48→384) | 38.6M | 2.015M | 2.081M | **1.19146** | −0.00998 (best) |
| 0136 raw FastMHL | ✗ | ✗ | 302.0M | 1.278M | 2.369M | 1.20567 | +0.00423 |
| **0138 output-compress-only** | ✗ | ✓ (48→384) | 38.2M | 1.130M | 1.193M | **1.21249** | +0.01105 (worst) |

## Findings
1. **The input compression (learned routing projection) is the essential component.** Removing it — routing
   on the raw 384-d input instead of a learned 48-d space — costs **+0.02103** (0121 → 0138), the largest
   swing on this line. The learned projection that *shapes the routing space* matters more than anything else.
2. **Decompress-only is actively harmful vs raw.** Adding output compression on top of raw 384-d routing makes
   it *worse*, not better: 0136 raw (1.20567) → 0138 decompress-only (1.21249), **+0.00682**. When the routing
   is already poor (raw 384-d), shrinking the tables to 48-wide + mixing removes the capacity the wide 384-d
   raw tables were using to partly compensate.
3. **Both projections together win.** 0121 (compress + decompress) is best and also *cheaper than raw* on
   params (38.6M vs 302M) — the compress→48-d→decompress bottleneck is doing real representational work, not
   just saving parameters.

## Cost note
0138 is the *cheapest* config on the line (66.91M total, FFN-FLOP 1.130M, vBW 1.193M — 12.5×/11.9× cheaper
than vanilla) because it drops the input compress matmul + weights while routing (coordinate-index gather) is
input-dim-independent. But it is also the *worst* on bpb — the cheapest ablation is not the useful one.

## Takeaway
Ranking: full CompressionMHL (1.19146) < raw FastMHL (1.20567) < output-compress-only (1.21249). The
**input routing projection carries the win**; output compression only helps when paired with it. For the
paper: the compress→LUT→decompress design is justified by the *input* projection first.
