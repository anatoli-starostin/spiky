#!/usr/bin/env bash
# exp_c31 — 3 seeds of PureLIF (TTFS) SAC on Walker2d, then the CPU reference for each.
#
# SEQUENTIAL, one GPU. Concurrency would only trade wall clock for contention: the
# membrane materialises a (batch, 192, 17, 17) tensor per step, so a single run already
# saturates the 5090.
#
# Each seed: train -> 100-episode deterministic CPU reference -> next seed. The eval is
# run inside the loop rather than batched at the end so a result exists for every finished
# seed even if a later one dies.
#
# Determinism flags match exp_c17/c30/c30b, so a rerun at a fixed seed reproduces bitwise.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for S in 0 1 2; do
  TAG="_c31_s${S}"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u pure_lif_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_pure_cpu.py "pure_lif_sac${TAG}_actor.npz" --episodes 100 \
      >> "cell_s${S}.log" 2>&1
done

touch SWEEP_DONE_C31
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
