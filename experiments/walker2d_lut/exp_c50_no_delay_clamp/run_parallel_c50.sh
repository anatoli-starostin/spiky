#!/usr/bin/env bash
# exp_c50 — 3 seeds of c49 with the delay clamp's LOWER bound removed.
#
# Byte-identical to exp_c49 except for one module constant: DELAY_MIN goes from 0.0 to
# -inf, so `clamp(delay, DELAY_MIN, t_window)` no longer has a causal floor. The upper
# t_window cap is KEPT -- it is what holds arrivals inside [., 2*t_window] so exp(a/tau)
# stays float32-safe in the reference's cumsum membrane, and removing it would confound
# the test with an overflow risk.
#
# WHAT THIS ISOLATES. c49 ends training with 94.6-94.9% of its 2,176 delays at or below
# the floor, where the clamp zeroes both the value AND the gradient -- permanently dead.
# c36, on the old unclamped module, ends ~40% negative and fully functional, spanning
# -10.08 to +12.67, and scores 4246 against c49's 2233. This run restores exactly that
# freedom and nothing else.
#
# Config comes from the trainer defaults (heads=1, tph=128, ndet=1, buckets=16,
# delay_init_std=0, delay_init_const=0, table_init_std=0.1 STOCK, share_betas=0,
# freeze_temperature=0); passing them as flags would only invite a transcription error in
# the least visible place.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local S=$1 TAG="_c50_s$1"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

for S in 0 1 2; do cell "$S" & done
wait

touch SWEEP_DONE_C50
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
