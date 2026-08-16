# exp086_compressionmhl_C10_6h_64_nodec_nap6_tied2x_4k

Sweep C run C10_6h_64_nodec_nap6: **tied** unembedder, **4096 steps**, **2x dense-FFN budget** (per-layer 2,359,296), **AdamW-LUT** (standard Sweep-B trainer: single AdamW, LUT tables in the no-wd group).

FFN slot: x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); n_heads=6, inner_in=64, inner_out=-1, nap=6, gamma=0, **tph=15**, independent per-head, hard.

Per-layer FFN slot params = compress 147,840 + lut 2,211,840 + decompress 0 = **2,359,680** (budget 2,359,296).

TOTAL = tied floor 16,131,840 + 6*2,359,680 = **30,289,920** (target 30,287,616, delta +2,304 = +0.0076%).

Reference to beat: tied dense baseline exp055 (4k) 1.35543; also compare to the Sweep-B AdamW-LUT 1x-budget run of the same shape.
