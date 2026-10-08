#!/bin/bash
# [lut_nanochat] REQUIRED smoke run for the DISTILLATION harness, before the long distill run.
# Mirrors runs/smoke_run_d24.sh but drives runs/distill_d24_1xh100.sh: it exercises the frozen-teacher load,
# the KL loss, checkpoint/resume and wandb, on the exact distillation code path, with its own run name / tag /
# results dir so nothing collides with the baseline or the real distill run.
#   phase A: 40-step horizon, --save-every 20, killed right after the step-20 checkpoint;
#   phase B: same launcher again -> auto-resume from step 20, finish at step 40, reduced base_eval + report;
#   check:   runs/smoke_check.py (distill mode: it also requires the KL metric to be present and dropping);
#   cleanup: deletes the smoke checkpoints + the symlink. The smoke wandb run is kept, marked -smoke / smoke tag.
# Usage (from experiments/lut_nanochat/nanochat):  bash runs/smoke_run_distill.sh      (or DBS=8 bash ...)

set -uo pipefail
cd "$(dirname "$0")/.."
export RUN_NAME=distill_d24_from_d24_1xh100-smoke
export MODEL_TAG=distill_d24_from_d24_1xh100_smoke
export RESULTS=$(cd .. && pwd)/results/$RUN_NAME
export NUM_ITERATIONS=40 SAVE_EVERY=20 EXTRA_TAGS=",smoke"
# Smoke: disable the in-training CORE + sampling, eval val bpb every 20 on a small split. The distill flags
# themselves live in the distill launcher's torchrun line; TRAIN_EXTRA only tweaks cadence (appended -> wins).
export TRAIN_EXTRA="--eval-every=20 --eval-tokens=2097152 --core-metric-every=-1 --sample-every=-1"
export EVAL_EXTRA="--max-per-task=16 --split-tokens=2097152"
export DBS=${DBS:-16}
export TEACHER_TAG=${TEACHER_TAG:-d24_1xh100}
export TEACHER_STEP=${TEACHER_STEP:-5568}
export DISTILL_T=${DISTILL_T:-1.0}
export DISTILL_ALPHA=${DISTILL_ALPHA:-0.0}
LAUNCHER=${LAUNCHER:-runs/distill_d24_1xh100.sh}
source runs/lut_env.sh
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

if [ -e "$CKPT_DIR" ] || [ -L "$CKPT_DIR" ] || [ -e "$RESULTS/train.log" ]; then
    echo "STOP: leftovers from a previous distill smoke run ($CKPT_DIR or $RESULTS). Move them away first:"
    echo "  rm -rf '$RESULTS'; rm -f '$CKPT_DIR'"
    exit 1
fi
mkdir -p "$RESULTS"

echo "== phase A: train to the step-20 checkpoint, then interrupt"
setsid bash "$LAUNCHER" > "$RESULTS/phaseA.out" 2>&1 &
PGID=$!
until grep -qE "^step 00021/" "$RESULTS/train.log" 2>/dev/null; do
    if ! kill -0 "$PGID" 2>/dev/null; then
        echo "phase A exited before step 21 (see $RESULTS/phaseA.out)"; break
    fi
    sleep 5
done
kill -TERM -- -"$PGID" 2>/dev/null || true
for _ in $(seq 60); do kill -0 "$PGID" 2>/dev/null || break; sleep 2; done
kill -KILL -- -"$PGID" 2>/dev/null || true
sleep 10
echo "phase A interrupted at: $(grep -E '^step [0-9]+/' "$RESULTS/train.log" | tail -1 | cut -c1-40)"

echo "== phase B: the same launcher command again (must auto-resume from step 20)"
bash "$LAUNCHER" > "$RESULTS/phaseB.out" 2>&1
echo "phase B exit: $?"

echo "== check"
source .venv/bin/activate
python runs/smoke_check.py --results "$RESULTS" --dbs "$DBS" --distill ${SMOKE_CHECK_EXTRA:-} | tee "$RESULTS/smoke_check.txt"
VERDICT=${PIPESTATUS[0]}

echo "== cleanup: deleting smoke checkpoints in $RESULTS/checkpoints (and the $CKPT_DIR symlink)"
rm -rf "$RESULTS/checkpoints"
if [ -L "$CKPT_DIR" ]; then rm -f "$CKPT_DIR"; else rm -rf "$CKPT_DIR"; fi
exit "$VERDICT"
