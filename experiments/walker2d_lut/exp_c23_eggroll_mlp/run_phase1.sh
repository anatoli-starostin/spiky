#!/usr/bin/env bash
# exp_c23 Phase 1 — EGGROLL vs full-rank ES on an MLP controller (#75).
#
# Two arms, identical in every respect except how the perturbation is generated:
#   A  EGGROLL, rank 4   (the method under test)
#   B  full-rank Gaussian ES == OpenES  (the control the paper itself compares against)
# Same seed, same population, same budget, same env, same evaluator. The paper's own RL
# finding is parity, so the pre-registered expectation is that these two land on top of
# each other while arm A samples 17x less noise per generation.
#
# Sequential on purpose: one GPU, and two concurrent MJX rollouts of this batch size
# would contend for it and make the wall-clock comparison meaningless.
#
# Determinism flags are the repo standard for this track (see exp_c17).
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Everything not named here is eggroll.py's default, which is the paper's tuned
# brax/ant EGGROLL column (Table 19): adam, lr 0.01, sigma 0.05, both decays 0.9995.
COMMON="--gens 300 --pop 1024 --episodes 2 --horizon 1000 --hidden 256 --layers 3 --seed 0"

echo "=== ARM A: EGGROLL rank 4   $(date -u +%FT%TZ) ==="
$PY -u eggroll.py --rank 4 $COMMON > arm_a_eggroll_r4.log 2>&1
echo "  rc=$? $(date -u +%FT%TZ)"

echo "=== ARM B: full-rank ES     $(date -u +%FT%TZ) ==="
$PY -u eggroll.py --rank 0 $COMMON > arm_b_fullrank.log 2>&1
echo "  rc=$? $(date -u +%FT%TZ)"

# The MJX fitness is a proxy. Every headline number comes from the CPU reference eval.
for t in eggroll_mlp256x3_r4_s0_theta.npz eggroll_mlp256x3_r0_s0_theta.npz; do
  if [ -f "$t" ]; then
    echo "=== CPU reference eval: $t  $(date -u +%FT%TZ) ==="
    $PY -u eval_cpu_eggroll.py "$t" >> cpu_evals.log 2>&1
    echo "  rc=$?"
  else
    echo "!! MISSING $t — its arm did not finish; check its log"
  fi
done

echo "PHASE 1 DONE $(date -u +%FT%TZ)"
