# exp_n_0029 — H8/d48/tph128, nap 6→7, tied, 16k

Clone of **exp_n_0004** (CompressionMultiHeadLUT FFN slot: H=8 heads, inner d=48, tph=128, tied,
vanilla backbone, 16k) with **ONLY nap 6 → 7** (2⁷=128 clusters per table instead of 64). This
roughly **doubles the LUT table parameters** (tables = H·tph·2^nap·d) but keeps **FLOPs the same**:
compress Linear(384→8·48) and decompress Linear(8·48→384) unchanged, routing is still cheap sign-tests
(now 7 anchor pairs). H·d = 8·48 = 384 → projection cost 2·384·384 = 294,912 MACs/layer, ~4× cheaper
than dense's 8·384² = 1,179,648 (unchanged from exp_n_0004).

**Params:** tables/layer = 8·128·2⁷·48 = 6,291,456 (was 3,145,728 at nap6). Per-layer = compress 147,840
+ decompress 147,840 + tables 6,291,456 = 6,587,136 → total = 16,131,840 + 6·6,587,136 = **55,654,656**
(≈ 2.40× dense's 23,209,728). Everything else identical to exp_n_0004.

Run 16k directly (per owner update — skip 4k scout). Compare to exp_n_0004 16k = 1.21738 and tied dense
16k exp073 = 1.19665. Question: do more clusters per table (finer routing) improve the LUT slot at fixed FLOPs?
