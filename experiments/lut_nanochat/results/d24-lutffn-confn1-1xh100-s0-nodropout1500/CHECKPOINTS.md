# Checkpoints: d24-lutffn-confn1-1xh100-s0-nodropout1500

Backed up from worker ds-h100-anatoly (spiky-lut_nanochat @ b38d0b33 checkout). Convention: `docs/CHECKPOINTS.md` in DeepSpike-AI/deepspike-ml. Bucket `deepspike-data`, run-name = results folder = wandb run name prefix.

- code commit: `437ba995` (spiky), launcher `runs/lut_ffn_d24_nodropout1500_1xh100.sh`; config: `results/d24-lutffn-confn1-1xh100-s0-nodropout1500/run_config.json` + `pin_config.json` where present, otherwise the `cmd:` line in `train.log`
- wandb run: `d24-lutffn-confn1-1xh100-s0-nodropout1500-8df9b1fa`
- nodropout1500: LUT-FFN v2, table dropout 0, 1501 steps; project `spiky-nanochat`
- files per step: `model_<step>.pt` (model state), `optim_<step>_rank0.pt` (optimizer state, 1 rank), `meta_<step>.json` (step, val_bpb, model/user config, dataloader state)
- how to load: copy a step's three files into `$NANOCHAT_BASE_DIR/base_checkpoints/d24_lutffn_confn1_1xh100_nodropout1500/` and run the launcher with `--resume-from-step=<step>` (see `nanochat/scripts/base_train.py`, `nanochat/scripts/base_eval.py` for eval); model-only: `torch.load(model_<step>.pt)` into the nanochat GPT built from `meta_<step>.json["model_config"]`

Upload verified 2026-11-08 (object count + exact byte size per object via `aws s3 ls --recursive`; sha256 column = local file; for the starred (*) files the S3 copy was streamed back and its sha256 is identical; the other files are verified by size only). Local copies are **not deleted**. Checkpoint files are never committed to git; this file only holds pointers.

| s3 path | what / comment | step | size | sha256 | loads with |
| --- | --- | --- | --- | --- | --- |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-nodropout1500/ckpt/step-001000/meta_001000.json | meta, intermediate, val_bpb 0.8987 | 1000 | 31.9 KB | ace78f8f044ccfc53c85a84245c7c4ccd3f7a2cc65f4d3ce642b73f6d33f5b4f | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-nodropout1500/ckpt/step-001500/meta_001500.json | meta, latest/final of this (truncated) run, val_bpb 0.8735 | 1500 | 31.9 KB | 976e3818b59df2dd4a553cc11885b7508a7471f929c35cde0dad7284337350b8 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-nodropout1500/ckpt/step-001000/model_001000.pt | model, intermediate, val_bpb 0.8987 | 1000 | 3.85 GB | df56d4ffeab194f2d527cf53c270b2841fd64db9c15e475dcdf61588533ada26 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-nodropout1500/ckpt/step-001500/model_001500.pt | model, latest/final of this (truncated) run, val_bpb 0.8735 | 1500 | 3.85 GB | a0517aff03c71f319bdfc7cf9bb0f7638d41ebec3986b7e8ba2b193afc84d887 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-nodropout1500/ckpt/step-001000/optim_001000_rank0.pt | optimizer state, intermediate, val_bpb 0.8987 | 1000 | 6.80 GB | 7ba1203f0a3eb3bc373c5046f9d625da414cad7f2785885d618c0e1bea3be666 | see above |
| s3://deepspike-data/anatoly/lut_nanochat/d24-lutffn-confn1-1xh100-s0-nodropout1500/ckpt/step-001500/optim_001500_rank0.pt | optimizer state, latest/final of this (truncated) run, val_bpb 0.8735 | 1500 | 6.80 GB | 0b8f3251e7e9457c335e0b7e761438afa18c659257d1d5e29eedf04901b36940 | see above |
