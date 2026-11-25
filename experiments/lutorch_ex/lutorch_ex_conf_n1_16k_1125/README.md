# lutorch_ex_conf_n1_16k — ConfidenceLUT n=1, exact 16K-step cosine

Same recipe as `../lutorch_ex_abl47_conf_n1_vm_1004_1806` (48K cosine) except `n_steps=16000` (warm-up fraction 0.1 -> 1600 steps; cosine to 0.1x completes at 16K).
`config.json` = reference config with n_steps=16000, exp_name changed. **`lut_read_top_n: 2` in it is NOT used for the cartridge**: `train_multi.py` with `CART=conf_n1` builds `ConfidenceLUT(read_top_n=1)` (same for the reference run); the key only feeds the discarded skeleton FFNs. `lut_light_forward_mode`/`lut_quant_mode` are likewise not used by this cartridge.

Launch (gpustar, RTX 5090): `CART=conf_n1 ABL47_CONFIG=config.json OUT_DIR=<dir> RUN_TAG=16k EXP_NAME=lutorch_ex_conf_n1_16k CKPT_EVERY=1000 python train_multi.py`
with the import overlay of `TRAIN_MULTI_LAUNCH.md` (spiky.lutorch from research/ffn_replacement_fix @ c4a750b2, lutorch_ex from this branch, == e45fd05e), tools from that branch, nanochat HEAD 92d63d4 (VM used da32e1d, not in upstream clone).
wandb: project Spiky, group lutorch_ex, entity anatoli-starostin-relocation. Checkpoint: s3://deepspike-data/anatoly/lutorch_ex/lutorch_ex_conf_n1_16k/ckpt.pt (model+opt, step 16000).

| step | 2000 | 4000 | 8000 | 12000 | 16000 |
|---|---|---|---|---|---|
| val bpb | 1.5093 | 1.3190 | 1.2180 | 1.1725 | 1.1522 |
