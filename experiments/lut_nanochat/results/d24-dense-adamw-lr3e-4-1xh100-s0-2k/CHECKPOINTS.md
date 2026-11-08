# Checkpoints: d24-dense-adamw-lr3e-4-1xh100-s0-2k

Backed up from worker ds-h100-anatoly (spiky-lut_nanochat @ b38d0b33 checkout). Convention: `docs/CHECKPOINTS.md` in DeepSpike-AI/deepspike-ml. Bucket `deepspike-data`, run-name = results folder = wandb run name prefix.

- code commit: `f6900340` (spiky), launcher `runs/dense_adamw_d24_2k_1xh100.sh`; config: `results/d24-dense-adamw-lr3e-4-1xh100-s0-2k/run_config.json` + `pin_config.json` where present, otherwise the `cmd:` line in `train.log`
- wandb run: `d24-dense-adamw-lr3e-4-1xh100-s0-2k-213e13da`
- first 2000 steps of the dense AdamW baseline (`--max-steps=2001`, NUM_ITERATIONS=5568 schedule); step 2000 is the resume point of the 5568-from2k run; `step-002000` files are also hardlinked into `smoke_results/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k-smoke/checkpoints/` (uploaded once, from this run)
- files per step: `model_<step>.pt` (model state), `optim_<step>_rank0.pt` (optimizer state, 1 rank), `meta_<step>.json` (step, val_bpb, model/user config, dataloader state)
- how to load: copy a step's three files into `$NANOCHAT_BASE_DIR/base_checkpoints/d24_dense_adamw_lr3e-4_2k/` and run the launcher with `--resume-from-step=<step>` (see `nanochat/scripts/base_train.py`, `nanochat/scripts/base_eval.py` for eval); model-only: `torch.load(model_<step>.pt)` into the nanochat GPT built from `meta_<step>.json["model_config"]`

Upload verified 2026-11-08 (object count + exact byte size per object via `aws s3 ls --recursive`; sha256 column = local file; for the starred (*) files the S3 copy was streamed back and its sha256 is identical; the other files are verified by size only). Local copies are **not deleted**. Checkpoint files are never committed to git; this file only holds pointers.

| s3 path | what / comment | step | size | sha256 | loads with |
| --- | --- | --- | --- | --- | --- |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-2k/ckpt/step-001500/meta_001500.json | meta, intermediate, val_bpb 0.8862 | 1500 | 31.8 KB | 93b4b8212f7146b479b42d6283486ba833e572f5ff6317b71ad62c194af6e824 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-2k/ckpt/step-002000/meta_002000.json | meta, resume point (used by 5568-from2k), val_bpb 0.8588 (hardlinked from smoke_results dir) | 2000 | 31.8 KB | b40450c756254c40c1eb7d4e1ce454cbc5b00d003c2fc42cf76fc16f19fff4f5 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-2k/ckpt/step-001500/model_001500.pt | model, intermediate, val_bpb 0.8862 | 1500 | 4.23 GB | 7dd3e918254a5fbbc900ceb677bfdb2704ef2e89ca52cad96ec07ac0a109fe1f | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-2k/ckpt/step-002000/model_002000.pt | model, resume point (used by 5568-from2k), val_bpb 0.8588 (hardlinked from smoke_results dir) | 2000 | 4.23 GB | 6fc7408d3937e21b91aa0e4333d2107593c79801a56efc8d37245487b4a0eed0 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-2k/ckpt/step-001500/optim_001500_rank0.pt | optimizer state, intermediate, val_bpb 0.8862 | 1500 | 8.46 GB | f81d6ea5a8c874e90e89aa115350c36de125be3acb0b81bfe8729bd081712b60 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-dense-adamw-lr3e-4-1xh100-s0-2k/ckpt/step-002000/optim_002000_rank0.pt | optimizer state, resume point (used by 5568-from2k), val_bpb 0.8588 (hardlinked from smoke_results dir) | 2000 | 8.46 GB | 647a1501771ce6436bea5fb53863197c6ddff6611e904615188b4f29870cf481 | see above |
