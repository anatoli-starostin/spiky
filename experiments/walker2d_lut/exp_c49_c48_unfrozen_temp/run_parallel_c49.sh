#!/usr/bin/env bash
# exp_c49 — 3 seeds of the n_det=1, 64-bucket config, co-resident on the 5090.
#
# The n_det=1 special case: ONE LIF per table, 64 ordered buckets, no mixed-radix
# combination at all. Same 64 cells/table as c38 (2^6) and c39 (4^3), so this is the pure
# "all width, one detector" end of the width-vs-count axis.
#
# Config comes from the trainer defaults (ndet=1, buckets=64, delay_init_std=4, no delay or
# boundary offset, table_init_std = 0.1/sqrt(32)); passing them as flags would only invite a
# transcription error in the least visible place.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local S=$1 TAG="_c49_s$1"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

for S in 0 1 2; do cell "$S" & done
wait

touch SWEEP_DONE_C49
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
