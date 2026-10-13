# 06_conf_n1_oldbias -- decompress-bias ablation of 06_conf_n1 (seed 1)

Causal test of whether the zero-init ProjectionMHL decompress bias (commit 955762e9) is responsible for the
~0.021 bpb edge of the fused conf_n1 over the abl47 reference-class conf_n1 (1.117706).

Identical to ../06_conf_n1 in EVERY way (same random_seed=1, lut_base_seed=1000, geometry, schedule, TV,
dropout, corpus, and train.py) EXCEPT: after model construction and before the optimizer, every block's
decompress.bias is re-initialised to the OLD (pre-955762e9) nn.Linear default U(+/-1/sqrt(384)) = U(+/-0.051031),
drawn from a DEDICATED torch.Generator(seed=777) so the GLOBAL RNG -- and therefore the dropout mask stream -- is
byte-identical to 06_conf_n1 seed 1. The decompress.bias VALUES are the only difference from that run.

exp_name = lutorch_ex_fused_abl47_conf_n1_oldbias (only config.json change). train.py = 06_conf_n1/train.py +
one guarded insertion (logs per-block bias L2/min/max/zero-count and asserts the global RNG is unchanged; ABORTs
if any block's bias is all-zero or the RNG moved). src/spiky/lutorch_ex is NOT modified.

Interpretation: if the final val_bpb jumps UP toward abl47's ~1.1177, the zero-init decompress bias is the cause of
the improvement. If it stays ~1.097 (like the 3 fused seeds), the bias init is NOT the cause.
