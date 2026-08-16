# exp082_compressionmhl_C8_6h_64-128_nap6_tied2x_4k

Sweep C run C8_6h_64-128_nap6: **tied** unembedder, **4096 steps**, **2x dense-FFN budget** (per-layer 2,359,296), **AdamW-LUT** (standard Sweep-B trainer: single AdamW, LUT tables in the no-wd group).

FFN slot: x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); n_heads=6, inner_in=64, inner_out=128, nap=6, gamma=0, **tph=39**, independent per-head, hard.

Per-layer FFN slot params = compress 147,840 + lut 1,916,928 + decompress 295,296 = **2,360,064** (budget 2,359,296).

TOTAL = tied floor 16,131,840 + 6*2,360,064 = **30,292,224** (target 30,287,616, delta +4,608 = +0.0152%).

Reference to beat: tied dense baseline exp055 (4k) 1.35543; also compare to the Sweep-B AdamW-LUT 1x-budget run of the same shape.
