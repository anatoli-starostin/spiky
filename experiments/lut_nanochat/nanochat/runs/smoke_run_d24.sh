#!/bin/bash
# [lut_nanochat] REQUIRED smoke run before the long baseline (RUNBOOK §7). ~20-30 min on 1xH100.
# It reuses runs/baseline_d24_1xh100.sh unchanged (same d24 config, fp8, FA3 pin, same DBS) through
# its default-off hooks, with its own run name / model tag / results dir, so nothing is shared with
# the real run:
#   phase A: launch with a 40-step horizon and --save-every 20; KILL it (SIGTERM to the whole process
#            group) once step 21 has started, i.e. right after the step-20 checkpoint was written;
#   phase B: rerun the SAME launcher command -> must auto-resume from step 20, finish at step 40, then run
#            a reduced standalone base_eval (16 examples/task, 2M tokens/split) and report_results.py;
#   check:   runs/smoke_check.py applies the PASS criteria mechanically and prints `SMOKE RUN: PASS|FAIL`;
#   cleanup: deletes the smoke checkpoints. The smoke wandb run is kept, marked by the `-smoke` name
#            suffix and the `smoke` tag (delete it in the wandb UI once the report is accepted).
# Usage (from experiments/lut_nanochat/nanochat):  bash runs/smoke_run_d24.sh      (or DBS=8 bash ...)

set -uo pipefail
cd "$(dirname "$0")/.."
export RUN_NAME=d24-dense-1xh100-s0-smoke
export MODEL_TAG=d24_1xh100_smoke
export RESULTS=$(cd .. && pwd)/results/$RUN_NAME
export NUM_ITERATIONS=40 SAVE_EVERY=20 EXTRA_TAGS=",smoke"
export TRAIN_EXTRA="--eval-every=20 --eval-tokens=2097152 --core-metric-every=-1 --sample-every=-1"
export EVAL_EXTRA="--max-per-task=16 --split-tokens=2097152"
export DBS=${DBS:-16}
LAUNCHER=${LAUNCHER:-runs/baseline_d24_1xh100.sh}     # overridable only for offline testing of this wrapper
source runs/lut_env.sh
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

if [ -e "$CKPT_DIR" ] || [ -e "$RESULTS/train.log" ]; then
    echo "STOP: leftovers from a previous smoke run ($CKPT_DIR or $RESULTS). Move them away first:"
    echo "  rm -rf '$CKPT_DIR' '$RESULTS'"
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
sleep 10                                                # let the GPU memory be released
echo "phase A interrupted at: $(grep -E '^step [0-9]+/' "$RESULTS/train.log" | tail -1 | cut -c1-40)"

echo "== phase B: the same launcher command again (must auto-resume from step 20)"
bash "$LAUNCHER" > "$RESULTS/phaseB.out" 2>&1
echo "phase B exit: $?"

echo "== check"
source .venv/bin/activate
python runs/smoke_check.py --results "$RESULTS" --dbs "$DBS" ${SMOKE_CHECK_EXTRA:-} | tee "$RESULTS/smoke_check.txt"
VERDICT=${PIPESTATUS[0]}

echo "== cleanup: deleting smoke checkpoints in $CKPT_DIR"
rm -rf "$CKPT_DIR"
exit "$VERDICT"
