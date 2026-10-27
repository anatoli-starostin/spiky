#!/bin/bash
# [lut_nanochat] First 500 steps of the v2 recipe with Muon-everywhere: whiten table optimizer + ProjectionMHL projections on Muon (fp8, --lut-ffn-proj-fp8-muon); attention/other block matrices are Muon as always.
# Experiment: experiments/lut_nanochat/results/d24-lutffn-confn1-1xh100-s0-whiten-projmuon500/README.md
#
# Identical to runs/lut_ffn_d24_whiten500_1xh100.sh plus --lut-ffn-proj-fp8-muon (projection GEMMs in fp8, weights on Muon at matrix-lr).
# Checkpoints at 250 and 500 are KEPT (KEEP_CKPTS=4) so the run can be continued to 5568.
#   bash runs/lut_ffn_d24_whiten_projmuon500_1xh100.sh            # real run  (wandb run d24-lutffn-confn1-1xh100-s0-whiten-projmuon500)
#   SMOKE=1 bash runs/lut_ffn_d24_whiten_projmuon500_1xh100.sh    # 12-step smoke (wandb tag "smoke", separate results/ckpt dir)
# Continue later (same launcher, resumes from the latest complete checkpoint):
#   TRAIN_EXTRA="--max-steps=5569" SAVE_EVERY=250 bash runs/lut_ffn_d24_whiten_projmuon500_1xh100.sh   (use a new RUN_NAME if it should be its own experiment)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${SMOKE:-0}" = "1" ]; then
    export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-1xh100-s0-whiten-projmuon500-smoke}
    export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_1xh100_whiten_projmuon500_smoke}
    export EXTRA_TAGS="${EXTRA_TAGS:-},smoke"
    export SAVE_EVERY=${SAVE_EVERY:-6}                # smoke also checks save + resume of the optimizer state
    export TRAIN_EXTRA="--eval-every=10 --core-metric-every=0 --max-steps=12 ${TRAIN_EXTRA:-}"
else
    export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-1xh100-s0-whiten-projmuon500}
    export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_1xh100_whiten_projmuon500}
    export EXTRA_TAGS="${EXTRA_TAGS:-},whiten-projmuon500"
    export SAVE_EVERY=${SAVE_EVERY:-250}
    export TRAIN_EXTRA="--max-steps=501 ${TRAIN_EXTRA:-}"
fi
export TRAIN_EXTRA="--lut-table-optimizer=whiten --lut-ffn-proj-fp8-muon ${TRAIN_EXTRA}"
export KEEP_CKPTS=${KEEP_CKPTS:-4}                 # keep steps 250, 500 (+ any later) for continuation
export NUM_ITERATIONS=${NUM_ITERATIONS:-5568}      # full v2 horizon: LR schedule identical to the full run
export SKIP_BASE_EVAL=1
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export LAUNCHER=runs/lut_ffn_d24_whiten_projmuon500_1xh100.sh
exec bash runs/lut_ffn_d24_1xh100.sh "$@"
