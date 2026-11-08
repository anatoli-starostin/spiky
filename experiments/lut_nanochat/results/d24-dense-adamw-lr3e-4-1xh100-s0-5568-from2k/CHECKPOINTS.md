# Checkpoints: d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k

Backed up from worker ds-h100-anatoly (spiky-lut_nanochat @ b38d0b33 checkout). Convention: `docs/CHECKPOINTS.md` in DeepSpike-AI/deepspike-ml. Bucket `deepspike-data`, run-name = results folder = wandb run name prefix.

- code commit: `55c50a0c` (spiky), launcher `runs/dense_adamw_d24_continue_1xh100.sh`; config: `results/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/run_config.json` + `pin_config.json` where present, otherwise the `cmd:` line in `train.log`
- wandb run: `d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k-94a95f93`
- continuation of `d24-dense-adamw-lr3e-4-1xh100-s0-2k` from its step 2000 (`--resume-from-step=2000`, dataloader resume approximate, see `resumes.log`); wandb project `anatoli-starostin-relocation/spiky-nanochat`. Final 5568 = the full-length dense AdamW baseline (CORE eval: `nanochat_base/base_eval/base_model_005568.csv`)
- files per step: `model_<step>.pt` (model state), `optim_<step>_rank0.pt` (optimizer state, 1 rank), `meta_<step>.json` (step, val_bpb, model/user config, dataloader state)
- how to load: copy a step's three files into `$NANOCHAT_BASE_DIR/base_checkpoints/d24_dense_adamw_lr3e-4_from2k/` and run the launcher with `--resume-from-step=<step>` (see `nanochat/scripts/base_train.py`, `nanochat/scripts/base_eval.py` for eval); model-only: `torch.load(model_<step>.pt)` into the nanochat GPT built from `meta_<step>.json["model_config"]`

Upload verified 2026-11-08 (object count + exact byte size per object via `aws s3 ls --recursive`; sha256 column = local file; for the starred (*) files the S3 copy was streamed back and its sha256 is identical; the other files are verified by size only). Local copies are **not deleted**. Checkpoint files are never committed to git; this file only holds pointers.

| s3 path | what / comment | step | size | sha256 | loads with |
| --- | --- | --- | --- | --- | --- |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/ckpt/final/model_005568.pt | model, **final**, val_bpb 0.7632 | 5568 | 4.23 GB | 65e56affeb4cbbd61f382492d72ee16e11ba98949f115da2b013a1f9f69db880* | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/ckpt/final/optim_005568_rank0.pt | optimizer state, **final**, val_bpb 0.7632 | 5568 | 8.46 GB | c9bfeca41e3475b004817c786663227f81cc1535ecd81907193a0efdccdd4f2a* | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/ckpt/final/meta_005568.json | meta, **final**, val_bpb 0.7632 | 5568 | 32.2 KB | 791ab9ec0d7315171edb62b0624608a90f6a7a2bc2aa1fa223964ea2204ed728 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/ckpt/step-005500/model_005500.pt | model, intermediate, val_bpb 0.7639 | 5500 | 4.23 GB | 007a93c08c3459b6b01789968bfe0d6762eacb319cd4747344cdaa5d83eff1ee | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/ckpt/step-005500/optim_005500_rank0.pt | optimizer state, intermediate, val_bpb 0.7639 | 5500 | 8.46 GB | 7655ca69787a5cad8ed83a6bc6a5614db4241e9fca2e3848bc4ac52bc44d7dc7 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/ckpt/step-005500/meta_005500.json | meta, intermediate, val_bpb 0.7639 | 5500 | 32.2 KB | 61c0dc484c370611026a20564e4fa2cab7820c39c0b156dc29a256269428bc7c | see above |
