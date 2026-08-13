#!/usr/bin/env bash
# exp_c41 — 3 seeds of MHL-LIF SAC, CONCURRENTLY on the one 5090.
#
# `XLA_PYTHON_CLIENT_PREALLOCATE=false` is what makes co-residency possible at all: with
# JAX's defaults the first process grabs 75% of the card (24.9 GB of 32.6) and the other
# two die instantly. That allocator default, not real demand, is why "1 GPU -> sequential"
# was the standing convention in this chapter.
#
# Expect ~1.5x aggregate throughput, not 3x, and not the 2.33x an earlier short-probe test
# suggested -- that measurement was inflated by JIT compile, which parallelises across CPU
# cores and dominates a short probe.
#
# Concurrency does not touch determinism: the processes are independent and each is still
# reproducible from its own seed under the XLA flags below.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local S=$1 TAG="_c41_s$1"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 \
      >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

cell 0 &
cell 1 &
cell 2 &
wait

touch SWEEP_DONE_C41
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
