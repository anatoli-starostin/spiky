# exp064_compressionmhl_B9_3h_96-96_nap6_g0_tied_4k

FFN-slot sweep run B9_3h_96-96_nap6_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=96, inner_out=96, nap=6, gamma=0, **tph=52**.

Per-layer FFN params: gamma_lin 0 + compress 110,880 + decompress 110,976 + lut 958,464 = **1,180,320** (budget 1,179,648, delta +672/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,180,320 = **23,213,760** (target 23,209,728, delta +4,032 = +0.0174%).
