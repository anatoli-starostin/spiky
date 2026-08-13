#!/usr/bin/env bash
# exp_c50b — six MORE seeds (3-8) of exp_c50, co-resident on the 5090.
#
# NOT a new configuration. Byte-identical to exp_c50: current unified LIFMultiHeadLUT,
# 1 head x 128 tables x 1 detector x 16 buckets, per-table betas, stock 0.1 table init,
# delay_init_std=0, SORT_FORM="rank", freeze_temperature=False, and the delay clamp with
# its LOWER bound removed (`clamp(delay, -inf, t_window)`) while the upper t_window cap is
# KEPT for float32 safety in the reference's cumsum membrane.
#
# WHY. c50 (seeds 0/1/2) settled the MECHANISM -- the learned delay distribution matched
# c36's seed for seed, across 6,528 parameters -- but not the RETURN: 3107.7 +/- 1728.7
# against c49's 2232.9 +/- 1259.1 is |t| 0.71, and against c36's 4246.1 +/- 298.4 is
# |t| 1.12. Two of three seeds recovered to within 5% of c36; one never took off. The c42b
# lesson is that a configuration failing about half the time shows 1/3 or 3/3 often enough
# to mislead, so the only honest way to read a takeoff rate here is more seeds. Seeds 3-8
# bring exp_c50 to n=9 pooled, the standard c42b established.
#
# The TAG stays `_c50_s{seed}` -- these ARE exp_c50, continued. Different seeds mean no
# filename collision with the 0/1/2 run, and pooling stays a glob rather than a rename.
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

for S in 3 4 5 6 7 8; do cell "$S" & done
wait

touch SWEEP_DONE_C50B
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
