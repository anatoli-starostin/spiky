# exp092 — C1 with inner 64/64 → 48/48 (param-matched), tied, 4k

Clone of Sweep-C **C1 = exp075** (tied, 6 heads, CompressionMHL, nap6, gamma0, AdamW-LUT,
4096 steps) with the inner compress/decompress dims **64/64 → 48/48**, and **tph re-solved to
stay param-matched** to the same Sweep-C 2× budget (per-layer 2,359,296; target ~30.29M).

## Param-match (inner=48/48, 6h, nap6 → tph=116)
- fixed = compress (6·48·385 = 110,880) + decompress (6·48·384+384 = 110,976) = **221,856**
- FastMHL budget = 2,359,296 − 221,856 = 2,137,440 → `tph = 2,137,440 / (6·2^6·48) = 115.96` → **tph = 116**
- per-layer = 221,856 + 6·116·64·48 (=2,138,112) = **2,359,968**
- total = 16,131,840 + 6·2,359,968 = **30,291,648** (target 30,287,616, +4,032 = +0.013%; smoke-confirmed)

Narrower inner (48 vs C1's 64) → more/wider... actually more tables per head (tph 84→116) at
the same budget. Tests whether a narrower addressed vector with more tables beats C1's 64/64.

References: C1/exp075 = 1.36613; tied dense exp055 (4k) = 1.35543. Everything else identical to
exp075 (6h, nap6, gamma0, tied, AdamW two-group, lr 3e-4, warmup+cosine, 4096 steps).
