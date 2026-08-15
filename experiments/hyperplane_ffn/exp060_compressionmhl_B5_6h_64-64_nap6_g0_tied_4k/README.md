# exp060_compressionmhl_B5_6h_64-64_nap6_g0_tied_4k

FFN-slot sweep run B5_6h_64-64_nap6_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=6, inner_in=64, inner_out=64, nap=6, gamma=0, **tph=36**.

Per-layer FFN params: gamma_lin 0 + compress 147,840 + decompress 147,840 + lut 884,736 = **1,180,416** (budget 1,179,648, delta +768/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,180,416 = **23,214,336** (target 23,209,728, delta +4,608 = +0.0199%).
