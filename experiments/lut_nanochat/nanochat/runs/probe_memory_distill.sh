#!/bin/bash
# [lut_nanochat] Pre-flight memory probe for DISTILLATION (~5-7 min): a few REAL d24 steps with the frozen
# teacher loaded and TWO forward passes per micro-step (student + teacher) + the KL, at the given device batch.
# Same decision rule as the baseline probe: peak RESERVED <= 75 GiB -> that batch is OK, else fall back.
#   bash runs/probe_memory_distill.sh          # probe batch 16
#   DBS=8 bash runs/probe_memory_distill.sh    # confirm the fallback (two full-vocab logits tensors are heavy)
# Uses the staged data/tokenizer and the trained teacher (model-tag $TEACHER_TAG @ $TEACHER_STEP); its own
# student model-tag and no wandb; deletes its checkpoint afterwards. Nothing here touches a real run's state.

set -uo pipefail
cd "$(dirname "$0")/.."
export RUN_NAME=${RUN_NAME:-distill_d24_from_d24_1xh100}
TEACHER_TAG=${TEACHER_TAG:-d24_1xh100}
TEACHER_STEP=${TEACHER_STEP:-5568}
DISTILL_T=${DISTILL_T:-1.0}
DISTILL_ALPHA=${DISTILL_ALPHA:-0.0}
source runs/lut_env.sh
source .venv/bin/activate
DBS=${DBS:-16}
LIMIT_GIB=${LIMIT_GIB:-75}
LOG="$RESULTS/probe_distill_dbs${DBS}.log"
STEPS=4    # steady-state memory is reached after step 1

timeout 1800 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size="$DBS" --fp8 \
    --distill-from="$TEACHER_TAG" --distill-step="$TEACHER_STEP" \
    --distill-temperature="$DISTILL_T" --distill-alpha="$DISTILL_ALPHA" \
    --num-iterations="$STEPS" --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --model-tag=probe_distill --run=dummy 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
rm -rf "$NANOCHAT_BASE_DIR/base_checkpoints/probe_distill"

RES=$(grep -oP 'Peak memory reserved: \K[0-9.]+' "$LOG" | tail -1)
ALLOC=$(grep -oP 'Peak memory usage: \K[0-9.]+' "$LOG" | tail -1)
STEP_MS=$(grep -oP 'step 0000[2-4]/.*?dt: \K[0-9.]+' "$LOG" | tail -1)
echo "------------------------------------------------------------"
echo "probe(distill) dbs=$DBS exit=$STATUS peak_allocated=${ALLOC:-?}MiB peak_reserved=${RES:-?}MiB step_time=${STEP_MS:-?}ms"
if [ "$STATUS" -eq 0 ] && [ -n "$RES" ] && python3 -c "import sys; sys.exit(0 if $RES/1024 <= $LIMIT_GIB else 1)"; then
    echo "VERDICT: distill batch $DBS OK (peak reserved $(python3 -c "print(f'{$RES/1024:.1f}')") GiB <= $LIMIT_GIB GiB) -> launch with DBS=$DBS"
elif grep -qiE "out of memory|OutOfMemoryError" "$LOG"; then
    echo "VERDICT: distill batch $DBS OOM -> fall back to a smaller DBS (raise grad-accum to keep the 2^20 global batch)"
elif [ -n "$RES" ]; then
    echo "VERDICT: distill batch $DBS too tight (peak reserved $(python3 -c "print(f'{$RES/1024:.1f}')") GiB > $LIMIT_GIB GiB) -> fall back to a smaller DBS"
else
    echo "VERDICT: distill probe FAILED for a non-memory reason (exit $STATUS) -> read $LOG; do not launch"
fi
