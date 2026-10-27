#!/bin/bash
# [lut_nanochat] Continuation of the dense AdamW d24 run (d24-dense-adamw-lr3e-4-1xh100-s0-2k, steps 0-2000) up to the
# full 5,568-step schedule. Separate experiment: own results folder / checkpoint dir / wandb run. The step-2000
# checkpoint of the first run is HARD-LINKED (no extra disk, the original stays untouched) into the new run's
# checkpoint dir, so the launcher's resume-from-latest picks it up. The new wandb run replays the first run's rows
# (step < 2000) so train/val/CORE curves are complete (--replay-wandb-from). Experiment README:
# results/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/README.md
#   SMOKE=1 bash runs/dense_adamw_d24_continue_1xh100.sh   # resume at 2000, 3 steps, replay included (wandb tag "smoke")
#   bash runs/dense_adamw_d24_continue_1xh100.sh           # continue to step 5568
set -euo pipefail
cd "$(dirname "$0")/.."

LR=${ADAMW_LR:-3e-4}
SRC_RUN=${SRC_RUN:-d24-dense-adamw-lr${LR}-1xh100-s0-2k}
SRC_RESULTS="$(cd .. && pwd)/results/$SRC_RUN"
SRC_WANDB_ID=$(cat "$SRC_RESULTS/wandb_run_id")
RESUME_STEP=${RESUME_STEP:-2000}
if [ "${SMOKE:-0}" = "1" ]; then
    export RUN_NAME=${RUN_NAME:-d24-dense-adamw-lr${LR}-1xh100-s0-5568-from2k-smoke}
    export MODEL_TAG=${MODEL_TAG:-d24_dense_adamw_lr${LR}_from2k_smoke}
    export EXTRA_TAGS="${EXTRA_TAGS:-},smoke,matrix_opt=adamw,adamw_lr=${LR},continued_from_2k"
    export SAVE_EVERY=${SAVE_EVERY:-100000}
    MAXARGS="--eval-every=2 --core-metric-every=0 --max-steps=$((RESUME_STEP + 3))"
else
    export RUN_NAME=${RUN_NAME:-d24-dense-adamw-lr${LR}-1xh100-s0-5568-from2k}
    export MODEL_TAG=${MODEL_TAG:-d24_dense_adamw_lr${LR}_from2k}
    export EXTRA_TAGS="${EXTRA_TAGS:-},matrix_opt=adamw,adamw_lr=${LR},continued_from_2k"
    export SAVE_EVERY=${SAVE_EVERY:-500}
    MAXARGS=""
fi
NEW_RESULTS="$(cd .. && pwd)/results/$RUN_NAME"
mkdir -p "$NEW_RESULTS/checkpoints"
# Seed the checkpoint dir with the resume step (hard links; skipped if already there, e.g. a crash-resume).
S=$(printf '%06d' "$RESUME_STEP")
for f in "model_$S.pt" "meta_$S.json" "optim_${S}_rank0.pt"; do
    [ -e "$NEW_RESULTS/checkpoints/$f" ] && continue
    [ -e "$SRC_RESULTS/checkpoints/$f" ] || { echo "STOP: $SRC_RESULTS/checkpoints/$f missing"; exit 1; }
    ln "$SRC_RESULTS/checkpoints/$f" "$NEW_RESULTS/checkpoints/$f"
done
export WANDB_GROUP=${WANDB_GROUP:-nanochat_baseline}
export ARCH_TAG=${ARCH_TAG:-dense}
export NUM_ITERATIONS=${NUM_ITERATIONS:-5568}
export TRAIN_EXTRA="--matrix-optimizer=adamw --adamw-matrix-lr=${LR} --replay-wandb-from=${WANDB_ENTITY:-anatoli-starostin-relocation}/${WANDB_PROJECT:-spiky-nanochat}/${SRC_WANDB_ID} ${MAXARGS} ${TRAIN_EXTRA:-}"
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export LAUNCHER=runs/dense_adamw_d24_continue_1xh100.sh
# Full run: no SKIP_BASE_EVAL, so the standalone base_eval + report run at step 5568 like the dense baseline.
[ "${SMOKE:-0}" = "1" ] && export SKIP_BASE_EVAL=1
exec bash runs/baseline_d24_1xh100.sh "$@"
