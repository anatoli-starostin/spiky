# exp076_compressionmhl_C2_6h_96-96_nap6_tied2x_4k

Sweep C run C2_6h_96-96_nap6: **tied** unembedder, **4096 steps**, **2x dense-FFN budget** (per-layer 2,359,296), **AdamW-LUT** (standard Sweep-B trainer: single AdamW, LUT tables in the no-wd group).

FFN slot: x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); n_heads=6, inner_in=96, inner_out=96, nap=6, gamma=0, **tph=52**, independent per-head, hard.

Per-layer FFN slot params = compress 221,760 + lut 1,916,928 + decompress 221,568 = **2,360,256** (budget 2,359,296).

TOTAL = tied floor 16,131,840 + 6*2,360,256 = **30,293,376** (target 30,287,616, delta +5,760 = +0.0190%).

Reference to beat: tied dense baseline exp055 (4k) 1.35543; also compare to the Sweep-B AdamW-LUT 1x-budget run of the same shape.
