# exp055_tied_vanilla_dense_ffn_baseline_4k

TIED-unembedder vanilla baseline for Sweep B: identical to exp032 (dense 384->1536->384 GELU FFN, 4096 steps) but with lm_head.weight = tok_emb.weight (weight sharing). This is Sweep B's reference val_bpb.

Params: tied non-FFN 16,131,840 + 6*dense-FFN 1,179,648 = **23,209,728** (target 23,209,728, delta +0).
