#!/usr/bin/env bash
# exp_c01 — SAC baseline on Walker2d-v5 (issue #75). Detached, persistent, unbuffered.
# Threads are pinned to 1: with torch's default thread pool the [256,256] MLP thrashes
# and throughput collapses to ~20 fps (measured). Pinned + CUDA gives ~230 fps.
set -euo pipefail
cd "$(dirname "$0")"

SEED="${1:-0}"
STEPS="${2:-1000000}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=glfw

PY=~/projects/spiky/.venv/bin/python
LOG="train_seed${SEED}.log"

echo "=== exp_c01 SAC seed=${SEED} steps=${STEPS} launch $(date -u +%FT%TZ) ===" > "$LOG"
"$PY" -u train_sac.py --seed "$SEED" --steps "$STEPS" --device cuda >> "$LOG" 2>&1
echo "=== EXIT=$? $(date -u +%FT%TZ) ===" >> "$LOG"
