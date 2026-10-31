#!/bin/bash
# Dominance-on-Confidence(n=1) follow-up: E1, E2, F sequentially on one GPU (4000-step compressed schedule, 1 seed).
set -o pipefail
cd "$(dirname "$0")"; source env.sh
export WANDB_API_KEY=$WANDB_SPIKY_API_KEY
mkdir -p runs
for A in E1_domconf_ln_fmh E2_domconf_ln_conf1 F_conf_n1; do
  CONFIG=configs/${A}_4k.json OUT_DIR=runs/${A}_4k python train.py 2>&1 | tee runs/${A}_4k.log || exit 1
done
