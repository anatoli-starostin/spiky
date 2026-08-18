# exp_n_0030 — H16, inner d=24, tph=64, nap6, tied, 16k

CompressionMultiHeadLUT FFN slot with **H=16 heads, inner d=24, tph=64, nap=6** (2⁶=64 clusters/table),
tied, vanilla backbone, 16k. Pushes the head count **past the H8–H12 saturation plateau** to H=16 with
narrower per-head d and fewer tables/head.

**H·d = 16·24 = 384** → stays on the fixed-throughput line: FFN projection cost 2·384·384 = 294,912
MACs/layer = ~4× cheaper than dense's 8·384² = 1,179,648 (same as exp_n_0004).

**Params (SMOKE-confirmed):** compress 16·24·385 = 147,840 + decompress 16·24·384+384 = 147,840 +
tables 16·64·2⁶·24 = 1,572,864 → per-layer 1,868,544 → total = 16,131,840 + 6·1,868,544 = **27,343,104**
(≈ 1.18× dense's 23,209,728 — a lean point). Everything else identical to exp_n_0004.

Runs 16k directly, serial after exp_n_0029. Compare to exp_n_0004 16k (H8/d48/tph128 = 1.21738),
exp_n_0005 (H12/d32/tph128 = 1.21739), and tied dense 16k (1.19665). Question: does H16 keep improving
past the H8–H12 plateau, or has head count fully saturated?
