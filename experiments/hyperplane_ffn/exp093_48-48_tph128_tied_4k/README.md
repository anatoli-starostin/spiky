# exp093 — inner 48/48 with tph 116→128 (power-of-two, over-budget), tied, 4k

Clone of **exp092** (C1 variant: tied, 6 heads, CompressionMHL inner_in=48/inner_out=48, nap6,
gamma0, AdamW-LUT, 4096 steps) with **ONLY tph changed 116 → 128** (a power of two).
Everything else identical.

**Intentionally over budget:** tph=128 (vs the param-matched 116). Per layer: compress 110,880
+ decompress 110,976 + tables (6·128·64·48 = 2,359,296) = **2,581,152** → total = 16,131,840 +
6·2,581,152 = **31,618,752** (+1,331,136 = **+4.4% over** the 30.29M 2× budget; +1.33M over
exp092). Not rescaled.

References: exp092 (inner 48/48, tph116) = 1.36168; C1/exp075 = 1.36613; tied dense exp055 =
1.35543. AdamW two-group optimizer, tied embeddings. One 4k run.
