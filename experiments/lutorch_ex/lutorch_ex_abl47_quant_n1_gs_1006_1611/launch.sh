#!/usr/bin/env bash
# Quantised Confidence n=1 (QuantisedConfidenceLUT, quant_mode="p2_int8", read_top_n=1), 48k steps, on gpustar.
# Harness: ../train_multi.py with CART=quant_n1. ABL47_CONFIG is the Quantised n=2 champion config
# (../lutorch_ex_abl47_quant_1004_1148/config.json), exactly as nebius's runs use it; CART switches the cartridge to
# read_top_n=1. ./config.json records the effective config: the champion with lut_read_top_n 2 -> 1, nothing else.
# Import layout: the spiky/ overlay described in ../TRAIN_MULTI_LAUNCH.md (old spiky.lutorch/util from
# research/ffn_replacement_fix @ c4a750b2, spiky.lutorch_ex from a copied snapshot of main @ e45fd05e).
#
# Launch AND resume: train_multi.py resumes from $OUT_DIR/ckpt.pt (every CKPT_EVERY steps), so re-running this script
# after a crash continues the run. Detached launch on gpustar (W&B offline inside the cage; `wandb sync` afterwards):
#   sbox bash -c 'setsid nohup <this dir>/launch.sh >> <this dir>/train.log 2>&1 < /dev/null & echo LAUNCHED'
set -eu
R=$HOME/projects/spiky
RUN=$R/experiments/lutorch_ex/lutorch_ex_abl47_quant_n1_gs_1006_1611
S=$HOME/projects/lx_quant_n1_scratch
VENV=$R/.venv
export CART=quant_n1 ABL47_CONFIG=$R/experiments/lutorch_ex/lutorch_ex_abl47_quant_1004_1148/config.json
export OUT_DIR=${OUT_DIR:-$RUN} CKPT_EVERY=${CKPT_EVERY:-1000} RUN_TAG=${RUN_TAG:-gs_1006_1611}
export PYTHONPATH=$S/xspiky_quant_n1
export TOOLS_DIR=$HOME/projects/spiky-ffnfix-tools/experiments/ffn_replacement/tools
export NANOCHAT_ROOT=$HOME/projects/nanochat
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export TRITON_CACHE_DIR=$HOME/.cache/triton TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_ext_quant_n1 MPLCONFIGDIR=/tmp/mpl
export LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib
[ -f "$HOME/.wandb_env" ] && . "$HOME/.wandb_env"
mkdir -p "$OUT_DIR" && cd "$OUT_DIR"
exec "$VENV/bin/python" -u "$R/experiments/lutorch_ex/train_multi.py"
