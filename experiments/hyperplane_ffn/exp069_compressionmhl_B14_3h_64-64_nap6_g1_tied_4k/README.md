# exp069_compressionmhl_B14_3h_64-64_nap6_g1_tied_4k

FFN-slot sweep run B14_3h_64-64_nap6_g1_tied (tied unembedder). x = x + 1*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=64, inner_out=64, nap=6, gamma=1, **tph=72**.

Per-layer FFN params: gamma_lin 147,840 + compress 73,920 + decompress 74,112 + lut 884,736 = **1,180,608** (budget 1,179,648, delta +960/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,180,608 = **23,215,488** (target 23,209,728, delta +5,760 = +0.0248%).
