#!/usr/bin/env bash
# exp_c42b — six NEW seeds (3..8) of the exp_c42 winning config, all co-resident.
#
# The config is byte-identical to exp_c42: standard i.i.d. half-normal delays
# (delay_init_std=4), NO delay offset, NO boundary offset, table_init_std = 0.1/sqrt(32).
# Those are the trainer's defaults, so no flags are passed -- passing them would invite a
# transcription error in the one place it would be least visible.
#
# Seeds 3..8 are DISJOINT from c42's 0,1,2, so the two runs pool to n=9 without reusing a
# single RNG stream. That is the whole point: 3/3 is consistent with a takeoff rate
# anywhere from ~0.4 upward, and only more seeds narrow it.
#
# SIX co-resident rather than two batches of three. Each process peaks at ~1,350 MiB, so
# six use ~8.1 GB of 32.6 -- comfortable. Aggregate throughput saturates well before six
# processes, so this finishes in about the same wall-clock as two sequential batches while
# needing one launch and one progress bar instead of two.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local S=$1 TAG="_c42b_s$1"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 \
      >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

for S in 3 4 5 6 7 8; do cell "$S" & done
wait

touch SWEEP_DONE_C42B
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
