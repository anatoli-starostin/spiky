# exp057_compressionmhl_B2_2h_64-64_nap6_g0_tied_4k

FFN-slot sweep run B2_2h_64-64_nap6_g0_tied (tied unembedder). x = x + 0*Linear(384->384)(h) + CompressionMultiHeadLUT(h), h=ln2(x); independent per-head (joint_head_compression=False), hard FastMHL.

Config: n_heads=2, inner_in=64, inner_out=64, nap=6, gamma=0, **tph=132**.

Per-layer FFN params: gamma_lin 0 + compress 49,280 + decompress 49,536 + lut 1,081,344 = **1,180,160** (budget 1,179,648, delta +512/layer).

TOTAL = tied non-FFN 16,131,840 + 6*1,180,160 = **23,212,800** (target 23,209,728, delta +3,072 = +0.0132%).
