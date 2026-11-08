# Checkpoints: d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000

Backed up from worker ds-h100-anatoly (spiky-lut_nanochat @ b38d0b33 checkout). Convention: `docs/CHECKPOINTS.md` in DeepSpike-AI/deepspike-ml. Bucket `deepspike-data`, run-name = results folder = wandb run name prefix.

- code commit: `db431524` (spiky), launcher `runs/lut_ffn_d24_whiten_projmuon1000_1xh100.sh`; config: `results/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/run_config.json` + `pin_config.json` where present, otherwise the `cmd:` line in `train.log`
- wandb run: `d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000-d07ba1ff`
- whiten tables + projection Muon (fp8), 1001 steps (truncated, schedule is for 5568); project `spiky-nanochat`
- files per step: `model_<step>.pt` (model state), `optim_<step>_rank0.pt` (optimizer state, 1 rank), `meta_<step>.json` (step, val_bpb, model/user config, dataloader state)
- how to load: copy a step's three files into `$NANOCHAT_BASE_DIR/base_checkpoints/d24_lutffn_confn1_1xh100_whiten_projmuon1000/` and run the launcher with `--resume-from-step=<step>` (see `nanochat/scripts/base_train.py`, `nanochat/scripts/base_eval.py` for eval); model-only: `torch.load(model_<step>.pt)` into the nanochat GPT built from `meta_<step>.json["model_config"]`

Upload verified 2026-11-08 (object count + exact byte size per object via `aws s3 ls --recursive`; sha256 column = local file; for the starred (*) files the S3 copy was streamed back and its sha256 is identical; the other files are verified by size only). Local copies are **not deleted**. Checkpoint files are never committed to git; this file only holds pointers.

| s3 path | what / comment | step | size | sha256 | loads with |
| --- | --- | --- | --- | --- | --- |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000250/meta_000250.json | meta, intermediate, val_bpb 1.0431 | 250 | 32.0 KB | 06097470022e4581e45656cfdc3e8be37ebad3375ed82eff17c185509c074305 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000500/meta_000500.json | meta, intermediate, val_bpb 0.9549 | 500 | 32.0 KB | 1df0d3ca6318b91a84ba7402be560002eedc9196ca2f635ea953db4e7a93e402 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000750/meta_000750.json | meta, intermediate, val_bpb 0.9158 | 750 | 32.0 KB | 4142681eee3f1968d415773908ef15d0d7577b528c78f3665b9df7b12136d300 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-001000/meta_001000.json | meta, latest/final of this (truncated) run, val_bpb 0.8942 | 1000 | 32.0 KB | b938e6d33536919e8cfc1b710e747d1e84a487e93781eff2a3a5450feb0d5883 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000250/model_000250.pt | model, intermediate, val_bpb 1.0431 | 250 | 3.85 GB | 2bfc152c515b300a0e06649f952ffd40fb10c490d0c07cadd4769fdef287407f | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000500/model_000500.pt | model, intermediate, val_bpb 0.9549 | 500 | 3.85 GB | 74d9004a1ba50f858996a7577af01a8a2779d524295e99e5b7397e81005b94ea | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000750/model_000750.pt | model, intermediate, val_bpb 0.9158 | 750 | 3.85 GB | 48e97b027c85041e77ffa1ccbf69207d47362ecb73bbff30b55586ad549aef7c | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-001000/model_001000.pt | model, latest/final of this (truncated) run, val_bpb 0.8942 | 1000 | 3.85 GB | c732a00a47c7466facef4b86d726ec5e6f6c51ddc749e85588a344e9434226f5* | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000250/optim_000250_rank0.pt | optimizer state, intermediate, val_bpb 1.0431 | 250 | 5.36 GB | 5f42cb435f54570742c4622f2c0a46560b8c3919703ca443ce0319afc57d39e9 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000500/optim_000500_rank0.pt | optimizer state, intermediate, val_bpb 0.9549 | 500 | 5.36 GB | ee85be2a971ad95229194bdc81345167fe4fb7e9971d7f79d152e937c47e8fc7 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-000750/optim_000750_rank0.pt | optimizer state, intermediate, val_bpb 0.9158 | 750 | 5.36 GB | d4787010770db0730e1ef71521b4e63a58bc8880158fecb1ebeda2fbedb749d6 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000/ckpt/step-001000/optim_001000_rank0.pt | optimizer state, latest/final of this (truncated) run, val_bpb 0.8942 | 1000 | 5.36 GB | dba2b368579e3ecd1e0d115b5d562fe05632ef33c63cbc4344ccdf8c367dc3b1* | see above |
