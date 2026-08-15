# exp056_compressionmhl_B1_1h_64-64_nap6_g0_tied_4k

FFN-slot sweep run B1_1h_64-64_nap6_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=1, inner_in=64, inner_out=64, nap=6, gamma=0, **tph=276**.

Per-layer FFN params: gamma_lin 0 + compress 24,640 + decompress 24,960 + lut 1,130,496 = **1,180,096** (budget 1,179,648, delta +448/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,180,096 = **23,212,416** (target 23,209,728, delta +2,688 = +0.0116%).
