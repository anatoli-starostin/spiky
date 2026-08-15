# exp044_compressionmhl_A4_4h_64-64_nap6_g0_4k

FFN-slot sweep run A4_4h_64-64_nap6_g0 (untied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=4, inner_in=64, inner_out=64, nap=6, gamma=0, **tph=60**.

Per-layer FFN params: gamma_lin 0 + compress 98,560 + decompress 98,688 + lut 983,040 = **1,180,288** (budget 1,179,648, delta +640/layer).

TOTAL = untied non-FFN 28,714,752 + 6*1,180,288 = **35,796,480** (target 35,792,640, delta +3,840 = +0.0107%).
