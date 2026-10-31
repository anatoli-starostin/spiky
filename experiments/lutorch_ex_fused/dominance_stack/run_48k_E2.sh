#!/bin/bash
# E2 (DominanceConf -> LN -> Conf n=1), 48000-step report-style run, 1 seed. Resumable: re-running continues from runs/E2_48k/ckpt.pt
# (saved every 2000 steps) and the same W&B run (fixed RUN_TAG + WANDB_RESUME=allow).
set -o pipefail
cd "$(dirname "$0")"; source env.sh
export WANDB_API_KEY=$WANDB_SPIKY_API_KEY WANDB_RESUME=allow RUN_TAG=main
mkdir -p runs
CONFIG=configs/E2_domconf_ln_conf1_48k.json OUT_DIR=runs/E2_48k python train.py 2>&1 | tee -a runs/E2_48k.log
