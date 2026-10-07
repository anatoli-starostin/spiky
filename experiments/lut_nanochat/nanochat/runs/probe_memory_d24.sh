#!/bin/bash
# [lut_nanochat] Pre-flight memory probe (~5 min): a few REAL d24 steps (fp8, compile, optimizer step)
# at the given device batch size, then a VERDICT line. Uses the staged data/tokenizer, its own model tag
# and no wandb; deletes its checkpoint afterwards. Nothing here touches the real run's state.
#   bash runs/probe_memory_d24.sh          # probe batch 16 (the default)
#   DBS=8 bash runs/probe_memory_d24.sh    # confirm the fallback
# Decision rule (our choice, not upstream's): peak RESERVED <= 75 GiB -> batch 16 OK
# (~4.5 GiB headroom on an 80 GB card for fragmentation and the eval passes); otherwise, or on OOM,
# use DBS=8 (64 grad-accum steps, identical maths).

set -uo pipefail
cd "$(dirname "$0")/.."
source runs/lut_env.sh
source .venv/bin/activate
DBS=${DBS:-16}
LIMIT_GIB=${LIMIT_GIB:-75}
LOG="$RESULTS/probe_dbs${DBS}.log"
STEPS=4    # 4 optimizer steps x (1048576 / (DBS*2048)) micro-steps; steady-state memory after step 1

timeout 1800 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size="$DBS" --fp8 \
    --num-iterations="$STEPS" --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --model-tag=probe_d24 --run=dummy 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
rm -rf "$NANOCHAT_BASE_DIR/base_checkpoints/probe_d24"

RES=$(grep -oP 'Peak memory reserved: \K[0-9.]+' "$LOG" | tail -1)
ALLOC=$(grep -oP 'Peak memory usage: \K[0-9.]+' "$LOG" | tail -1)
STEP_MS=$(grep -oP 'step 0000[2-4]/.*?dt: \K[0-9.]+' "$LOG" | tail -1)
echo "------------------------------------------------------------"
echo "probe dbs=$DBS exit=$STATUS peak_allocated=${ALLOC:-?}MiB peak_reserved=${RES:-?}MiB step_time=${STEP_MS:-?}ms"
if [ "$STATUS" -eq 0 ] && [ -n "$RES" ] && python3 -c "import sys; sys.exit(0 if $RES/1024 <= $LIMIT_GIB else 1)"; then
    echo "VERDICT: batch $DBS OK (peak reserved $(python3 -c "print(f'{$RES/1024:.1f}')") GiB <= $LIMIT_GIB GiB) -> launch with DBS=$DBS"
elif grep -qiE "out of memory|OutOfMemoryError" "$LOG"; then
    echo "VERDICT: batch $DBS OOM -> fall back to batch 8 (DBS=8, 64 grad-accum steps, identical maths)"
elif [ -n "$RES" ]; then
    echo "VERDICT: batch $DBS too tight (peak reserved $(python3 -c "print(f'{$RES/1024:.1f}')") GiB > $LIMIT_GIB GiB) -> fall back to batch 8 (DBS=8)"
else
    echo "VERDICT: probe FAILED for a non-memory reason (exit $STATUS) -> read $LOG; do not launch"
fi
