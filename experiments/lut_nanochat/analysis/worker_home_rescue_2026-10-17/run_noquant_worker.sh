#!/bin/bash
# Worker-VM launcher for the non-quant champion lutorch_ex reproduction (full weight transfer).
# Verifies bit-exact (script aborts if max|Δ|>2e-3) THEN trains 48k. Online wandb. Run detached.
set -euo pipefail
export VP=/home/nebius-h100/venv/bin/python
export PYTHONPATH=/home/nebius-h100/projects/spiky/src
export TOOLS_DIR=/home/nebius-h100/projects/spiky/experiments/ffn_replacement/tools
export NANOCHAT_ROOT=/home/nebius-h100/projects/nanochat
export NANOCHAT_BASE_DIR=/home/nebius-h100/.cache/nanochat
export NOQUANT_CONFIG=/home/nebius-h100/projects/spiky/experiments/ffn_replacement/lut_ablation/exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph64_h8_seed1_noquant_headdrop20/config.json
export MPLCONFIGDIR=/tmp/mpl
export CUDA_VISIBLE_DEVICES=0
export VERIFY_ONLY=0
export N_STEPS=48000
export EVAL_EVERY=500
export OUT_DIR=/home/nebius-h100/out_noquant
export RUN_NAME=lutorch_ex_noquant_champion_xfer
export WANDB_GROUP=lut_ablation
export WANDB_BASE_URL=https://api.wandb.ai
"$VP" /home/nebius-h100/repro_noquant_champion.py
