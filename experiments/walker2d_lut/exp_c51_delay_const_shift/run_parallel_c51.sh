#!/usr/bin/env bash
# exp_c51 — 3 seeds of c49 with the delays initialised INSIDE the clamp window.
#
# The other half of the c50/c51 pair. The clamp stays exactly as upstream wrote it
# (delay_min = 0.0, cap = t_window); what changes is where training starts.
# delay_init_const = 3.2 puts every delay 3.2 units off the floor -- 3.2 is the mean of
# the half-normal that delay_init_std=4 would have drawn, so this is the same central
# tendency as c38-c47 used, with none of the variance.
#
# WHAT THIS ISOLATES. If the trap is fatal only because delay_init_std=0 puts every delay
# exactly ON the boundary -- where the first update pushes roughly half of them below zero
# and they can never return -- then simply starting in the interior should recover the
# return WITHOUT touching the module. That is the cheap, upstream-compatible fix, and it
# is the one worth preferring if it works: c50 changes the model, c51 changes one number.
#
# Config comes from the trainer defaults (heads=1, tph=128, ndet=1, buckets=16,
# delay_init_std=0, delay_init_const=3.2, table_init_std=0.1 STOCK, share_betas=0,
# freeze_temperature=0); passing them as flags would only invite a transcription error in
# the least visible place.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local S=$1 TAG="_c51_s$1"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

for S in 0 1 2; do cell "$S" & done
wait

touch SWEEP_DONE_C51
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
