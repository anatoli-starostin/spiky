#!/bin/bash
# [lut_nanochat] d24 with the LUT-FFN drop-in on ONE H100 80GB.
#
# Identical recipe to runs/baseline_d24_1xh100.sh (d24, --target-param-data-ratio=8, --fp8, ClimbMix,
# global batch 2^20, 5,568 steps) EXCEPT every transformer block's dense MLP is replaced by the locked
# ConfidenceLUT FFN (fp32 island): ProjectionMHL(ConfidenceLUT(LUTSpec h_in=h_out=16, d_in=d_out=48,
# tph=64, nap=8, anchor_mode="pairs"), seed=1, read_top_n=1, beta=2.0, gamma=1.0, learnable_score=True,
# weight_init_std=1e-3, table_dropout=0.2) with decompress.{weight,bias} zero-init (clean identity drop-in),
# cell-TV lambda=10, LUT params trained with AdamW (NOT Muon). ~358.7M LUT-FFN params vs 453M dense FFN
# -> whole model ~1.290B (dense d24 is 1.384B). See ../README.md "Stage 2" and base_train --lut-ffn flags.
#
# This is a THIN WRAPPER over baseline_d24_1xh100.sh: it reuses all the checkpoint / resume / prune /
# base_eval machinery and only overrides the run identity, wandb grouping, arch tag, and the --lut-ffn flags.
#
# Usage (from experiments/lut_nanochat/nanochat):
#   bash runs/lut_ffn_d24_1xh100.sh            # fresh start, or resume from the latest checkpoint
#   DBS=8 bash runs/lut_ffn_d24_1xh100.sh      # smaller device batch if 16 does not fit (LUT adds fp32 island memory)
# NOT launched automatically. The locked geometry lives in base_train's --lut-ffn* defaults; override via TRAIN_EXTRA.

set -euo pipefail
cd "$(dirname "$0")/.."

# Run identity + wandb grouping (own results dir + own wandb run id, separate from the dense baseline)
export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-1xh100-s0}
export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_1xh100}
export WANDB_GROUP=${WANDB_GROUP:-nanochat_lut_ffn}

# Arch tag + the LUT-FFN switch. Locked geometry = base_train --lut-ffn* defaults (h=16,d=48,tph=64,nap=8,
# read_top_n=1,beta=2,gamma=1,seed=1,std=1e-3,table_dropout=0.2,tv_lambda=10,lut_lr=3e-3).
export ARCH_TAG=${ARCH_TAG:-lut_ffn_confidence_n1}
export EXTRA_TAGS="${EXTRA_TAGS:-},ffn=confidenceLUT,read_top_n=1,r=768,tph=64,nap=8,cells=256"
export TRAIN_EXTRA="--lut-ffn ${TRAIN_EXTRA:-}"

exec bash runs/baseline_d24_1xh100.sh "$@"
