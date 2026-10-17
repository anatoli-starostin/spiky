#!/bin/bash
# Worker-VM launcher for the matched-init QUANT champion reproduction (p2_int8, full weight transfer).
# Verifies bit-exact (aborts if max|Δ|>2e-3) THEN trains 48k. Online wandb. Run detached. Parallel to the float job.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH          # nvcc for the OLD pow2_int8 + NEW lutorch_ex quant kernel JIT builds
export CUDA_HOME=/usr/local/cuda
export VP=/home/nebius-h100/venv/bin/python
"$VP" -m pip install -q ninja >/dev/null 2>&1 || true   # torch cpp_extension JIT needs ninja
export PYTHONPATH=/home/nebius-h100/projects/spiky/src
export TOOLS_DIR=/home/nebius-h100/projects/spiky/experiments/ffn_replacement/tools
export NANOCHAT_ROOT=/home/nebius-h100/projects/nanochat
export NANOCHAT_BASE_DIR=/home/nebius-h100/.cache/nanochat
export QUANT_CONFIG=/home/nebius-h100/p2int8_config.json
export TRITON_CACHE_DIR=/home/nebius-h100/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/nebius-h100/.cache/inductor_lutorch_ex
export MPLCONFIGDIR=/tmp/mpl
export CUDA_VISIBLE_DEVICES=0
export VERIFY_ONLY=0
export N_STEPS=48000
export EVAL_EVERY=500
export OUT_DIR=/home/nebius-h100/out_quant
export RUN_NAME=lutorch_ex_quant_champion_xfer
export WANDB_GROUP=lut_ablation
export WANDB_BASE_URL=https://api.wandb.ai
"$VP" /home/nebius-h100/repro_quant_champion.py
