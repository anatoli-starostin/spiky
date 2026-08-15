# exp052_compressionmhl_A12_3h_64_nod_nap6_g0_4k

FFN-slot sweep run A12_3h_64_nod_nap6_g0 (untied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=64, inner_out=-1, nap=6, gamma=0, **tph=15**.

Per-layer FFN params: gamma_lin 0 + compress 73,920 + decompress 0 + lut 1,105,920 = **1,179,840** (budget 1,179,648, delta +192/layer).

TOTAL = untied non-FFN 28,714,752 + 6*1,179,840 = **35,793,792** (target 35,792,640, delta +1,152 = +0.0032%).
