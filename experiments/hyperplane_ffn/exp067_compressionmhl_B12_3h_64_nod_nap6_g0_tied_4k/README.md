# exp067_compressionmhl_B12_3h_64_nod_nap6_g0_tied_4k

FFN-slot sweep run B12_3h_64_nod_nap6_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=64, inner_out=-1, nap=6, gamma=0, **tph=15**.

Per-layer FFN params: gamma_lin 0 + compress 73,920 + decompress 0 + lut 1,105,920 = **1,179,840** (budget 1,179,648, delta +192/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,179,840 = **23,210,880** (target 23,209,728, delta +1,152 = +0.0050%).
