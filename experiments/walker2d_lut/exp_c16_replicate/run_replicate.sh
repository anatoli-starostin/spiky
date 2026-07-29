#!/usr/bin/env bash
# exp_c16 — THE REPLICATE TEST: same config, same seed, twice, sequentially (#75).
#
# The question every number in exp_c12..exp_c15 depends on: when two runs differ, is that
# the SEED, or is it run-to-run nondeterminism? Everything so far has varied the seed and
# called the result "seed-sd" without ever checking that a fixed seed reproduces.
#
# Config is exp_c15's verbatim -- hyperplane x hard, nap6/tph32/heads1, torch-faithful
# anchor_pairs init -- with seed 0 BOTH times. The only difference between run A and run B
# is that they are two separate invocations.
#
# SEQUENTIAL, one at a time, so each run has the GPU to itself. That is deliberate on two
# counts: it removes contention as a confound, and it matches how exp_c11 ran (26.7
# min/cell with the GPU to itself) -- exp_c11 being one half of the unexplained same-seed
# discrepancy this test is meant to bottom out.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--addressing hyperplane --hyperplane-init anchor_pairs \
        --hyperplane-anchor-policy canonical_full_coverage \
        --forward-mode hard --nap 6 --tph 32 --heads 1 --seed 0 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

for rep in a b; do
  echo "=== replicate $rep (seed 0, GPU to itself)  $(date -u +%FT%TZ) ==="
  $PY -u "$TRAIN" $COMMON --tag "_c16_rep_${rep}" > "cell_rep_${rep}.log" 2>&1
  echo "  rc=$?  $(date -u +%FT%TZ)"
done

echo "BOTH REPLICATES DONE $(date -u +%FT%TZ)"
