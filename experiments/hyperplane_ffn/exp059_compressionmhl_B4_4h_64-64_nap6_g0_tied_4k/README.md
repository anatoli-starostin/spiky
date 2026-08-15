# exp059_compressionmhl_B4_4h_64-64_nap6_g0_tied_4k

FFN-slot sweep run B4_4h_64-64_nap6_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=4, inner_in=64, inner_out=64, nap=6, gamma=0, **tph=60**.

Per-layer FFN params: gamma_lin 0 + compress 98,560 + decompress 98,688 + lut 983,040 = **1,180,288** (budget 1,179,648, delta +640/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,180,288 = **23,213,568** (target 23,209,728, delta +3,840 = +0.0165%).
