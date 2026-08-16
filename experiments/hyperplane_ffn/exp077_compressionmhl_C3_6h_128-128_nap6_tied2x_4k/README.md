# exp077_compressionmhl_C3_6h_128-128_nap6_tied2x_4k

Sweep C run C3_6h_128-128_nap6: **tied** unembedder, **4096 steps**, **2x dense-FFN budget** (per-layer 2,359,296), **AdamW-LUT** (standard Sweep-B trainer: single AdamW, LUT tables in the no-wd group).

FFN slot: x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); n_heads=6, inner_in=128, inner_out=128, nap=6, gamma=0, **tph=36**, independent per-head, hard.

Per-layer FFN slot params = compress 295,680 + lut 1,769,472 + decompress 295,296 = **2,360,448** (budget 2,359,296).

TOTAL = tied floor 16,131,840 + 6*2,360,448 = **30,294,528** (target 30,287,616, delta +6,912 = +0.0228%).

Reference to beat: tied dense baseline exp055 (4k) 1.35543; also compare to the Sweep-B AdamW-LUT 1x-budget run of the same shape.
