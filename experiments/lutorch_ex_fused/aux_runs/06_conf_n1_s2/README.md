# 06_conf_n1_s2 — seed-spread replicate of 06_conf_n1

Identical to ../06_conf_n1 (same fused cartridge FusedConfidenceLUT read_top_n=1, geometry
h8/tph64/nap8/d48 r=384, 6-block MinimalGPT, 48k steps, lr 3e-4 cosine, wd 0.1, grad_accum 4,
device_batch 12, TV lambda 10, table_dropout 0.2, corpus ClimbMix, online-or-die wandb) EXCEPT:

- random_seed = 2   (original 06_conf_n1 used random_seed = 1)
- exp_name    = lutorch_ex_fused_abl47_conf_n1_s2   (for a distinct wandb run name)

train.py is BYTE-IDENTICAL to 06_conf_n1/train.py (md5 6ebb49ecfd0b96e145a64e3dcdc8389a); only config.json differs, and only
in those two fields. lut_base_seed is held at 1000 (as in the original) so the
spread is driven by the torch-RNG seed — backbone init (nn.init on the global RNG) and the table-dropout
mask stream — which is the hypothesised source of the run-to-run val_bpb variation.

Purpose: measure the seed spread of the final val_bpb against the first run's 1.096980, to judge whether
the ~0.02 bpb edge over the abl47 reference-class conf_n1 (1.117706 final) is a real gain or seed noise.
Launched as a BARE `python train.py` (online-or-die wandb). NOT via sbox.
