#!/usr/bin/env bash
# exp_c17 — DETERMINISM TEST: the exp_c16 replicate, with deterministic GPU ops (#75).
#
# exp_c16 showed |A - B| = 999.1 at an identical seed. Prime suspect: the table-weight
# scatter-add in the custom_vjp backward, which accumulates through GPU atomics in
# nondeterministic order. This reruns the identical replicate with determinism forced on.
#
# The real test is NOT the eval return -- it is whether the two CHECKPOINTS are bit-for-bit
# identical. Returns can coincide by luck; 28M weights cannot.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
# --xla_gpu_deterministic_ops forces deterministic kernels (notably the atomics-based
# scatter/reduce paths). The cuBLAS workspace pin is the documented companion: without a
# fixed workspace, cuBLAS may pick different split-k reductions between runs.
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

COMMON="--addressing hyperplane --hyperplane-init anchor_pairs \
        --hyperplane-anchor-policy canonical_full_coverage \
        --forward-mode hard --nap 6 --tph 32 --heads 1 --seed 0 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

echo "XLA_FLAGS=$XLA_FLAGS"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

for rep in a b; do
  echo "=== deterministic replicate $rep (seed 0, GPU to itself)  $(date -u +%FT%TZ) ==="
  $PY -u "$TRAIN" $COMMON --tag "_c17_det_${rep}" > "cell_det_${rep}.log" 2>&1
  echo "  rc=$?  $(date -u +%FT%TZ)"
done

echo "BOTH DETERMINISTIC RUNS DONE $(date -u +%FT%TZ)"
