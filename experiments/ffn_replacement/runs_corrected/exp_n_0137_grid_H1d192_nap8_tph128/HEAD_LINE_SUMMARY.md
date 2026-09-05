# Head line @67.35M complete — H1/d192 endpoint (exp_n_0137)

The H1·d192 point (CompressionMHL n_heads=1, inner=192, nap8/tph128; H·d=192 held → 67.35M, table 37.7M)
finishes the iso-param head line. **final val_bpb 1.19448** (best 1.19432).

## The full head line (67.35M, H·d=192 fixed, nap8/tph128) — U-shaped, minimum at H2
| config | val_bpb | Δ vs untied vanilla (1.20144) |
|---|---|---|
| H1 · d192 (0137) | 1.19448 | −0.00696 |
| **H2 · d96 (0131)** | **1.18883** | **−0.01261** (best) |
| H4 · d48 (0121, anchor) | 1.19146 | −0.00998 |
| H8 · d24 (0132) | 1.19263 | −0.00881 |

**The head line is not monotone — it turns over.** The earlier reading "fewer/wider heads win"
(H2 < H4 < H8) does **not** extend to the single-head extreme: **H1 (one wide d192 head) is the *worst* of
the four**, and **H2/d96 is the sweet spot**. So there is an interior optimum at 2 heads:
- too few heads (H1): a single routing head gives no routing diversity — worst.
- H2: enough routing multiplicity while each head stays wide (d96) — best.
- more/narrower heads (H4→H8): each head's routing space shrinks (d48→d24) and quality erodes again.

All four beat untied vanilla (1.20144). H1 (1.19448) is the lowest-FFN-FLOP head-line point (routing FLOP
minimal at H=1) but pays for it in bpb.

## Takeaway
Head count is a *diversity vs per-head-width* trade with an optimum at **H2/d96**, not a monotone
"fewer is better". Update to the earlier claim: the win is at a small-but-nonzero head count, and the
single-head degenerate case is the worst on this line.
