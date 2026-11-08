# Checkpoints: d24-lutffn-confn1-1xh100-s0-whiten500

Backed up from worker ds-h100-anatoly (spiky-lut_nanochat @ b38d0b33 checkout). Convention: `docs/CHECKPOINTS.md` in DeepSpike-AI/deepspike-ml. Bucket `deepspike-data`, run-name = results folder = wandb run name prefix.

- code commit: `3627c60a` (spiky), launcher `runs/lut_ffn_d24_whiten500_1xh100.sh`; config: `results/d24-lutffn-confn1-1xh100-s0-whiten500/run_config.json` + `pin_config.json` where present, otherwise the `cmd:` line in `train.log`
- wandb run: `d24-lutffn-confn1-1xh100-s0-whiten500-c57246a7`
- whiten table optimizer, 501 steps; project `spiky-nanochat`
- files per step: `model_<step>.pt` (model state), `optim_<step>_rank0.pt` (optimizer state, 1 rank), `meta_<step>.json` (step, val_bpb, model/user config, dataloader state)
- how to load: copy a step's three files into `$NANOCHAT_BASE_DIR/base_checkpoints/d24_lutffn_confn1_1xh100_whiten500/` and run the launcher with `--resume-from-step=<step>` (see `nanochat/scripts/base_train.py`, `nanochat/scripts/base_eval.py` for eval); model-only: `torch.load(model_<step>.pt)` into the nanochat GPT built from `meta_<step>.json["model_config"]`

Upload verified 2026-11-08 (object count + exact byte size per object via `aws s3 ls --recursive`; sha256 column = local file; for the starred (*) files the S3 copy was streamed back and its sha256 is identical; the other files are verified by size only). Local copies are **not deleted**. Checkpoint files are never committed to git; this file only holds pointers.

| s3 path | what / comment | step | size | sha256 | loads with |
| --- | --- | --- | --- | --- | --- |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten500/ckpt/step-000250/meta_000250.json | meta, intermediate, val_bpb 1.0538 | 250 | 31.9 KB | 00fae0eeabc8da60535ee55f4d4f01d95397cc7f5b9f62d984de7f82b2f993b9 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten500/ckpt/step-000500/meta_000500.json | meta, latest/final of this (truncated) run, val_bpb 0.9588 | 500 | 31.9 KB | 587b6fd07a8e7edf3456aee9febd925801c9bcbc43464fc457ddffe3b52ca8c7 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten500/ckpt/step-000250/model_000250.pt | model, intermediate, val_bpb 1.0538 | 250 | 3.85 GB | 698bda8dba630601882a9ed8ca5cd53dbdfc9f1750be6610cea6866e02fff76d | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten500/ckpt/step-000500/model_000500.pt | model, latest/final of this (truncated) run, val_bpb 0.9588 | 500 | 3.85 GB | 6e3d85ab4bb89580a48a21f4a3d68a63b9b606f7f2bc13c6b6dc2c5bb6ee6ad0 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten500/ckpt/step-000250/optim_000250_rank0.pt | optimizer state, intermediate, val_bpb 1.0538 | 250 | 5.59 GB | 8dd6bb2b1091e6f3ec40f60407828b075b6a7f071744c41055f1bdf6f889f81e | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-whiten500/ckpt/step-000500/optim_000500_rank0.pt | optimizer state, latest/final of this (truncated) run, val_bpb 0.9588 | 500 | 5.59 GB | da774efc6e16d536d583d3765b0fb3cbfc4624565e1c8ba2115758529a112cb7 | see above |
