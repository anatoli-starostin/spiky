# exp043_compressionmhl_A2_2h_64-64_nap6_g0_4k

FFN-slot sweep run A2_2h_64-64_nap6_g0 (untied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=2, inner_in=64, inner_out=64, nap=6, gamma=0, **tph=132**.

Per-layer FFN params: gamma_lin 0 + compress 49,280 + decompress 49,536 + lut 1,081,344 = **1,180,160** (budget 1,179,648, delta +512/layer).

TOTAL = untied non-FFN 28,714,752 + 6*1,180,160 = **35,795,712** (target 35,792,640, delta +3,072 = +0.0086%).
