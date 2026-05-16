#!/usr/bin/env bash
# Run all out_proj sweep sub-experiments sequentially, size-ascending.
# Reads ordered list from configs.txt; each sub-dir's stdout -> stdout.log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP_DIR="$REPO_ROOT/transformer_exps/sweep_out_proj"
PY="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"

mapfile -t CONFIGS < "$SWEEP_DIR/configs.txt"
TOTAL=${#CONFIGS[@]}

echo "==> Sweep start: $(date -Iseconds)  ($TOTAL configs)"
i=0
for cfg in "${CONFIGS[@]}"; do
    i=$((i+1))
    sub="$SWEEP_DIR/$cfg"
    log="$sub/stdout.log"
    echo "==> [$i/$TOTAL $cfg] start: $(date -Iseconds)"
    if "$PY" -u "$sub/train.py" > "$log" 2>&1; then
        echo "==> [$i/$TOTAL $cfg] done:  $(date -Iseconds)"
    else
        echo "==> [$i/$TOTAL $cfg] FAILED at $(date -Iseconds) — see $log"
    fi
done
echo "==> Sweep end: $(date -Iseconds)"
