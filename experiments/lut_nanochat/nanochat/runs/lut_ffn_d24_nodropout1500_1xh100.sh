#!/bin/bash
# [lut_nanochat] LUT-FFN v2 recipe with table dropout OFF, first 1500 steps of the 5,568-step schedule, one H100.
# Experiment: experiments/lut_nanochat/results/d24-lutffn-confn1-1xh100-s0-nodropout1500/README.md
# Only differences vs runs/lut_ffn_d24_repro500_1xh100.sh: --lut-ffn-table-dropout=0 (v2: 0.2), --max-steps=1501, SAVE_EVERY=500.
# 1501 (not 1500) so val/CORE/checkpoint at step 1500 happen at the top of the loop (see the repro500 launcher).
#   bash runs/lut_ffn_d24_nodropout1500_1xh100.sh            # real run
#   SMOKE=1 bash runs/lut_ffn_d24_nodropout1500_1xh100.sh    # 12-step smoke (wandb tag "smoke")
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${SMOKE:-0}" = "1" ]; then
    export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-1xh100-s0-nodropout1500-smoke}
    export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_1xh100_nodropout1500_smoke}
    export EXTRA_TAGS="${EXTRA_TAGS:-},smoke,table_dropout=0"
    export SAVE_EVERY=${SAVE_EVERY:-100000}
    export TRAIN_EXTRA="--lut-ffn-table-dropout=0 --eval-every=10 --core-metric-every=0 --max-steps=12 ${TRAIN_EXTRA:-}"
else
    export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-1xh100-s0-nodropout1500}
    export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_1xh100_nodropout1500}
    export EXTRA_TAGS="${EXTRA_TAGS:-},nodropout1500,table_dropout=0"
    export SAVE_EVERY=${SAVE_EVERY:-500}
    export TRAIN_EXTRA="--lut-ffn-table-dropout=0 --max-steps=1501 ${TRAIN_EXTRA:-}"
fi
export NUM_ITERATIONS=${NUM_ITERATIONS:-5568}      # full v2 horizon: LR/WSD schedule identical to the full run
export SKIP_BASE_EVAL=1
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export LAUNCHER=runs/lut_ffn_d24_nodropout1500_1xh100.sh
exec bash runs/lut_ffn_d24_1xh100.sh "$@"
