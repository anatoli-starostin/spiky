# exp054_compressionmhl_A14_3h_64-64_nap6_g1_4k

FFN-slot sweep run A14_3h_64-64_nap6_g1 (untied unembedder). x = x + 1*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=64, inner_out=64, nap=6, gamma=1, **tph=72**.

Per-layer FFN params: gamma_lin 147,840 + compress 73,920 + decompress 74,112 + lut 884,736 = **1,180,608** (budget 1,179,648, delta +960/layer).

TOTAL = untied non-FFN 28,714,752 + 6*1,180,608 = **35,798,400** (target 35,792,640, delta +5,760 = +0.0161%).
