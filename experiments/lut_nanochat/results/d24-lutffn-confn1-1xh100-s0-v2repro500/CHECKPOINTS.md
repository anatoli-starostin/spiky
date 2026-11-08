# Checkpoints: d24-lutffn-confn1-1xh100-s0-v2repro500

Backed up from worker ds-h100-anatoly (spiky-lut_nanochat @ b38d0b33 checkout). Convention: `docs/CHECKPOINTS.md` in DeepSpike-AI/deepspike-ml. Bucket `deepspike-data`, run-name = results folder = wandb run name prefix.

- code commit: `3c53f98a` (spiky), launcher `runs/lut_ffn_d24_repro500_1xh100.sh`; config: `results/d24-lutffn-confn1-1xh100-s0-v2repro500/run_config.json` + `pin_config.json` where present, otherwise the `cmd:` line in `train.log`
- wandb run: `d24-lutffn-confn1-1xh100-s0-v2repro500-2959ed04`
- v2 repro, 501 steps; project `spiky-nanochat`
- files per step: `model_<step>.pt` (model state), `optim_<step>_rank0.pt` (optimizer state, 1 rank), `meta_<step>.json` (step, val_bpb, model/user config, dataloader state)
- how to load: copy a step's three files into `$NANOCHAT_BASE_DIR/base_checkpoints/d24_lutffn_confn1_1xh100_v2repro500/` and run the launcher with `--resume-from-step=<step>` (see `nanochat/scripts/base_train.py`, `nanochat/scripts/base_eval.py` for eval); model-only: `torch.load(model_<step>.pt)` into the nanochat GPT built from `meta_<step>.json["model_config"]`

Upload verified 2026-11-08 (object count + exact byte size per object via `aws s3 ls --recursive`; sha256 column = local file; for the starred (*) files the S3 copy was streamed back and its sha256 is identical; the other files are verified by size only). Local copies are **not deleted**. Checkpoint files are never committed to git; this file only holds pointers.

| s3 path | what / comment | step | size | sha256 | loads with |
| --- | --- | --- | --- | --- | --- |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-v2repro500/ckpt/step-000250/meta_000250.json | meta, intermediate, val_bpb 1.0560 | 250 | 31.7 KB | 3cc8c290806bd0fb6041f8ed027a2ef370bc72a2742ffc2de42c9635b5670f5d | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-v2repro500/ckpt/step-000500/meta_000500.json | meta, latest/final of this (truncated) run, val_bpb 0.9603 | 500 | 31.7 KB | e7be64f088341d0d166e48fbd74405c5022bf71c8c00135d2e2889c62e8c16f0 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-v2repro500/ckpt/step-000250/model_000250.pt | model, intermediate, val_bpb 1.0560 | 250 | 3.85 GB | caa3ebbab603db3952ac5f72048abf21f1439e700c1e13cb90df1e0353a4e00d | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-v2repro500/ckpt/step-000500/model_000500.pt | model, latest/final of this (truncated) run, val_bpb 0.9603 | 500 | 3.85 GB | 7e08086eb8691d1ea80e2fdecb02c09ac38d9ace1e6605ea66c0e8a43d80b23f | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-v2repro500/ckpt/step-000250/optim_000250_rank0.pt | optimizer state, intermediate, val_bpb 1.0560 | 250 | 6.80 GB | c03434f97b1f62c4fe99b6351e6ff01852902c37b1508718b6f18a2feff77e4f | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-v2repro500/ckpt/step-000500/optim_000500_rank0.pt | optimizer state, latest/final of this (truncated) run, val_bpb 0.9603 | 500 | 6.80 GB | f4b8b23734e836f5d69c25b71bb3235fbdd070287260fe456aa6372c2e8aeb30 | see above |
