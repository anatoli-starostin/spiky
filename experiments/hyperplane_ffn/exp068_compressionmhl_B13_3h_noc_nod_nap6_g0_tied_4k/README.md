# exp068_compressionmhl_B13_3h_noc_nod_nap6_g0_tied_4k

FFN-slot sweep run B13_3h_noc_nod_nap6_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=3, inner_in=-1, inner_out=-1, nap=6, gamma=0, **tph=16**.

Per-layer FFN params: gamma_lin 0 + compress 0 + decompress 0 + lut 1,179,648 = **1,179,648** (budget 1,179,648, delta +0/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,179,648 = **23,209,728** (target 23,209,728, delta +0 = +0.0000%).
