# exp094 — 12 heads, inner 48/48, tph=64 (fixed), tied, 4k

Clone of **exp093** (tied, CompressionMHL inner_in=48/inner_out=48, nap6, gamma0, AdamW-LUT,
4096 steps) with **n_heads 6→12** and **tph→64** (fixed power of two, NOT re-solved for
budget). Doubles the head count vs the 6 we standardized on.

**Not on budget (intended):** per layer compress (12·48·385 = 221,760) + decompress
(12·48·384+384 = 221,568) + tables (12·64·64·48 = 2,359,296) = **2,802,624** → total =
16,131,840 + 6·2,802,624 = **32,947,584** (+2,659,968 = **+8.8% over** the 30.29M 2× budget).
Not rescaled.

References: exp093 (6h 48/48 tph128) = 1.36062 (current 6-head 48/48 best); C1/exp075 (6h
64/64) = 1.36613; tied dense exp055 = 1.35543. Everything else identical to exp093 (inner
48/48, nap6, gamma0, tied, AdamW two-group). One 4k run — tests whether doubling heads (12)
helps at fixed inner=48.
