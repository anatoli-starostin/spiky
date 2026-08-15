# exp061_compressionmhl_B6_3h_64-64_nap5_g0_tied_4k

FFN-slot sweep run B6_3h_64-64_nap5_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=64, inner_out=64, nap=5, gamma=0, **tph=168**.

Per-layer FFN params: gamma_lin 0 + compress 73,920 + decompress 74,112 + lut 1,032,192 = **1,180,224** (budget 1,179,648, delta +576/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,180,224 = **23,213,184** (target 23,209,728, delta +3,456 = +0.0149%).
