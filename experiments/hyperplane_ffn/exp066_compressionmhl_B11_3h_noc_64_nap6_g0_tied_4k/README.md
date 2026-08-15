# exp066_compressionmhl_B11_3h_noc_64_nap6_g0_tied_4k

FFN-slot sweep run B11_3h_noc_64_nap6_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=-1, inner_out=64, nap=6, gamma=0, **tph=90**.

Per-layer FFN params: gamma_lin 0 + compress 0 + decompress 74,112 + lut 1,105,920 = **1,180,032** (budget 1,179,648, delta +384/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,180,032 = **23,212,032** (target 23,209,728, delta +2,304 = +0.0099%).
