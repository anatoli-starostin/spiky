# exp048_compressionmhl_A8_3h_32-32_nap6_g0_4k

FFN-slot sweep run A8_3h_32-32_nap6_g0 (untied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=32, inner_out=32, nap=6, gamma=0, **tph=180**.

Per-layer FFN params: gamma_lin 0 + compress 36,960 + decompress 37,248 + lut 1,105,920 = **1,180,128** (budget 1,179,648, delta +480/layer).

TOTAL = untied non-FFN 28,714,752 + 6*1,180,128 = **35,795,520** (target 35,792,640, delta +2,880 = +0.0080%).
