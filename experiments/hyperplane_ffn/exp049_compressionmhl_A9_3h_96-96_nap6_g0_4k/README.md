# exp049_compressionmhl_A9_3h_96-96_nap6_g0_4k

FFN-slot sweep run A9_3h_96-96_nap6_g0 (untied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=96, inner_out=96, nap=6, gamma=0, **tph=52**.

Per-layer FFN params: gamma_lin 0 + compress 110,880 + decompress 110,976 + lut 958,464 = **1,180,320** (budget 1,179,648, delta +672/layer).

TOTAL = untied non-FFN 28,714,752 + 6*1,180,320 = **35,796,672** (target 35,792,640, delta +4,032 = +0.0113%).
