# S2a nap9, tph 128→64 — full 16k (exp_n_0120) — the tph cliff (task c1958230)

exp_n_0119 (nap9 tph128) with ONE knob change: **tph 128 → 64**. train.py byte-identical; full 16k,
effective batch 24,576, schedule identical to 0118/0119. Third point on the tph trace 256→128→64.

## The tph trace (nap9, H4 in48 out48; predicted == measured)
| exp | tph | params | FFN FLOPs | vBW | final bpb | vs 0084 | vs 0118 |
|---|---|---|---|---|---|---|---|
| 0118 | 256 | 180.60M | 2.286M | 2.375M | 1.17460 | −0.02406 | 0 |
| 0119 | 128 | 105.10M | 2.028M | 2.081M | 1.18386 | −0.01480 | +0.00926 |
| **0120** | **64** | **67.35M** | **1.898M** | **1.933M** | **1.19859** | **−0.00007** | **+0.02399** |
| 0084 (dense-V) | — | 67.35M | 2.236M | 2.375M | 1.19866 | 0 | +0.02406 |
| exp_n_0045 tied | — | — | — | — | 1.1977 | | 0120 vs tied = **+0.00089** |

## Headline — the nap9 win is a PARAM-SCALE win, not an architectural free lunch
At tph64 the routed FFN uses **the exact same param/table budget as dense-V exp_n_0084** (67.35M total,
37.7M table — nap9×tph64 = nap7×tph256 in cell×table count) and lands in a **dead heat: 1.19859 vs
1.19866 (Δ−0.00007)**, and marginally *behind* the tied baseline (+0.0009). So the entire −0.024 advantage
0118 held over dense-V came from **spending the extra ~113M params on a bigger table** — at iso-params the
routed FFN gives essentially nothing over dense-V.

## The penalty ACCELERATES (cliff), not sub-linear
Marginal cost of each halving:
- 256→128: **+0.00926** bpb for −75.5M params
- 128→64: **+0.01473** bpb for −37.7M params

The second halving costs *more bpb while removing fewer params* — the win/param falls off steeply at the
low end. Win over dense-V: −0.0241 → −0.0148 → −0.00007 (collapses to zero at 0084's budget).

## Step-aligned: the tph64 gap GROWS with training
Gap vs 0118 (tph256): 0119 (tph128) was a *flat* ~+0.0094 offset, but **0120 (tph64) starts at +0.019
early and widens to +0.024 late** — the big table keeps paying off more as training proceeds (capacity
matters more with more tokens). So tph64 doesn't just start behind; it falls further behind.

## Takeaway
Table count (tph) is the knob that carries nap9's win, and it's **super-linear at the top / cliff at the
bottom**: tph256 (180M) is clearly best, tph128 (105M) keeps ~62% of the win for 42% fewer params, but
tph64 (67M, = dense-V budget) erases the win entirely. If the goal is beating dense-V, you must pay in
table params; the routed FFN is not a free architectural win at iso-params. Sweet spot for the
quality/param trade is around tph128–192; tph64 is not worth it.

See `tph_trace_256_128_64.png`.
