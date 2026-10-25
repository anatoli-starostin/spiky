#!/bin/bash
# [lut_nanochat] Reproduce the FIRST 500 STEPS of d24-lutffn-confn1-1xh100-s0-full-v2 on one H100.
# Experiment: experiments/lut_nanochat/results/d24-lutffn-confn1-1xh100-s0-v2repro500/README.md
#
# Runs the unchanged v2 recipe (runs/lut_ffn_d24_1xh100.sh: LUT-FFN, --no-lr-rescale, full 5,568-step LR/WSD
# horizon) but stops early with --max-steps. 501 (not 500) so the loop reaches the top of step 500: val bpb,
# CORE and the step-500 checkpoint happen there (they are skipped if the loop breaks at 500).
#   bash runs/lut_ffn_d24_repro500_1xh100.sh            # real run  (wandb run d24-lutffn-confn1-1xh100-s0-v2repro500)
#   SMOKE=1 bash runs/lut_ffn_d24_repro500_1xh100.sh    # 12-step smoke (wandb tag "smoke", separate results/ckpt dir)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${SMOKE:-0}" = "1" ]; then
    export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-1xh100-s0-v2repro500-smoke}
    export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_1xh100_v2repro500_smoke}
    export EXTRA_TAGS="${EXTRA_TAGS:-},smoke"
    MAXSTEPS=12
    export SAVE_EVERY=${SAVE_EVERY:-100000}
    export TRAIN_EXTRA="--eval-every=10 --core-metric-every=0 --max-steps=$MAXSTEPS ${TRAIN_EXTRA:-}"
else
    export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-1xh100-s0-v2repro500}
    export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_1xh100_v2repro500}
    export EXTRA_TAGS="${EXTRA_TAGS:-},repro500"
    export SAVE_EVERY=${SAVE_EVERY:-250}
    export TRAIN_EXTRA="--max-steps=501 ${TRAIN_EXTRA:-}"
fi
export NUM_ITERATIONS=${NUM_ITERATIONS:-5568}      # the full v2 horizon: LR schedule identical to the full run
export SKIP_BASE_EVAL=1                            # no full-horizon checkpoint exists; the 500-step evals run in-training
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export LAUNCHER=runs/lut_ffn_d24_repro500_1xh100.sh
exec bash runs/lut_ffn_d24_1xh100.sh "$@"
