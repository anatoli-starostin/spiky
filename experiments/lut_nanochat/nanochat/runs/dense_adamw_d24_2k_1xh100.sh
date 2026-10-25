#!/bin/bash
# [lut_nanochat] Dense d24 baseline with AdamW instead of Muon for the transformer matrices, first 2000 steps of the
# full 5,568-step schedule, one H100. Fork of d24-dense-1xh100-s0 (runs/baseline_d24_1xh100.sh): same model, data,
# fp8, batch, seed, LR rescale (ON, as in the dense baseline) and WSD schedule; ONLY the matrix optimizer changes.
# Experiment: results/d24-dense-adamw-lr<LR>-1xh100-s0-2k/README.md
#   ADAMW_LR=1e-3 bash runs/dense_adamw_d24_2k_1xh100.sh      # base LR of the matrix AdamW (default 1e-3)
#   SMOKE=1 bash runs/dense_adamw_d24_2k_1xh100.sh            # 12-step smoke (wandb tag "smoke")
set -euo pipefail
cd "$(dirname "$0")/.."

LR=${ADAMW_LR:-1e-3}
if [ "${SMOKE:-0}" = "1" ]; then
    export RUN_NAME=${RUN_NAME:-d24-dense-adamw-lr${LR}-1xh100-s0-2k-smoke}
    export MODEL_TAG=${MODEL_TAG:-d24_dense_adamw_lr${LR}_2k_smoke}
    export EXTRA_TAGS="${EXTRA_TAGS:-},smoke,matrix_opt=adamw,adamw_lr=${LR}"
    export SAVE_EVERY=${SAVE_EVERY:-100000}
    MAXARGS="--eval-every=10 --core-metric-every=0 --max-steps=12"
else
    export RUN_NAME=${RUN_NAME:-d24-dense-adamw-lr${LR}-1xh100-s0-2k}
    export MODEL_TAG=${MODEL_TAG:-d24_dense_adamw_lr${LR}_2k}
    export EXTRA_TAGS="${EXTRA_TAGS:-},adamw2k,matrix_opt=adamw,adamw_lr=${LR}"
    export SAVE_EVERY=${SAVE_EVERY:-500}
    MAXARGS="--max-steps=2001"                      # 2001: val/CORE/checkpoint at step 2000 run at the top of the loop
fi
export WANDB_GROUP=${WANDB_GROUP:-nanochat_baseline}   # same group as the dense Muon baseline
export ARCH_TAG=${ARCH_TAG:-dense}
export NUM_ITERATIONS=${NUM_ITERATIONS:-5568}          # full horizon: LR/WSD schedule as in the 5.5K-step run
export TRAIN_EXTRA="--matrix-optimizer=adamw --adamw-matrix-lr=${LR} ${MAXARGS} ${TRAIN_EXTRA:-}"
export SKIP_BASE_EVAL=1
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export LAUNCHER=runs/dense_adamw_d24_2k_1xh100.sh
exec bash runs/baseline_d24_1xh100.sh "$@"
