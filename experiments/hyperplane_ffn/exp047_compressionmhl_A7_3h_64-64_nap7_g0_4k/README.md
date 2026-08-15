# exp047_compressionmhl_A7_3h_64-64_nap7_g0_4k

FFN-slot sweep run A7_3h_64-64_nap7_g0 (untied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=64, inner_out=64, nap=7, gamma=0, **tph=42**.

Per-layer FFN params: gamma_lin 0 + compress 73,920 + decompress 74,112 + lut 1,032,192 = **1,180,224** (budget 1,179,648, delta +576/layer).

TOTAL = untied non-FFN 28,714,752 + 6*1,180,224 = **35,796,096** (target 35,792,640, delta +3,456 = +0.0097%).
