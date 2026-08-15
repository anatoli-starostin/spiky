# exp050_compressionmhl_A10_3h_128-128_nap6_g0_4k

FFN-slot sweep run A10_3h_128-128_nap6_g0 (untied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=128, inner_out=128, nap=6, gamma=0, **tph=36**.

Per-layer FFN params: gamma_lin 0 + compress 147,840 + decompress 147,840 + lut 884,736 = **1,180,416** (budget 1,179,648, delta +768/layer).

TOTAL = untied non-FFN 28,714,752 + 6*1,180,416 = **35,797,248** (target 35,792,640, delta +4,608 = +0.0129%).
