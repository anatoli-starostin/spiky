# exp078_compressionmhl_C4_6h_192-192_nap6_tied2x_4k

Sweep C run C4_6h_192-192_nap6: **tied** unembedder, **4096 steps**, **2x dense-FFN budget** (per-layer 2,359,296), **AdamW-LUT** (standard Sweep-B trainer: single AdamW, LUT tables in the no-wd group).

FFN slot: x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); n_heads=6, inner_in=192, inner_out=192, nap=6, gamma=0, **tph=20**, independent per-head, hard.

Per-layer FFN slot params = compress 443,520 + lut 1,474,560 + decompress 442,752 = **2,360,832** (budget 2,359,296).

TOTAL = tied floor 16,131,840 + 6*2,360,832 = **30,296,832** (target 30,287,616, delta +9,216 = +0.0304%).

Reference to beat: tied dense baseline exp055 (4k) 1.35543; also compare to the Sweep-B AdamW-LUT 1x-budget run of the same shape.
