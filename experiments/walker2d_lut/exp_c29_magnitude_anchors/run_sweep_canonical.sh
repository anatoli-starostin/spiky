#!/usr/bin/env bash
# exp_c29, WAVE 2 — the identical sweep under lutorch's CANONICAL_FULL_COVERAGE anchor
# sampler instead of BALANCED (#75).
#
# WHY. `balanced` balances which COORDINATES get used; `canonical_full_coverage`
# balances which PAIRS get used, and repairs within-table duplicates. Measured at this
# exact shape (64 tables x 6 bits, seeds 0/1/2, d=17): balanced covers 94-97% of the 136
# possible comparators while reusing others up to 8 times and leaves 2-6 tables holding
# a redundant bit, so a table addresses 61.5 of 64 rows on average; canonical covers
# 100%, reuses each 2-3 times, and reaches 64.0/64 every seed. It is also the policy
# FastMultiHeadLut actually ships with -- it REJECTS `balanced`. This wave asks whether
# that better-conditioned draw changes any conclusion.
#
# SAME DIRECTORY, DIFFERENT TAG, deliberately. The two waves must differ in ONE flag and
# nothing else, and the surest way to guarantee that is to share the trainer, the
# evaluator, the instrumentation and -- above all -- constants.json, rather than to copy
# them into a second directory where they could drift. Wave 1 writes lut_sac_c29_*,
# wave 2 writes lut_sac_c29c_*; the log prefixes are cell_ and cellc_. Nothing collides.
#
# WAITS FOR WAVE 1. Starts only once run_sweep.sh has written SWEEP_DONE, which it does
# after its own CPU evals, so the two waves never contend for the GPU. Bounded: if wave 1
# dies without a sentinel this aborts loudly instead of waiting forever.
set -u
cd "$(dirname "$0")"

MAXWAIT=$((8 * 3600))
waited=0
while [ ! -f SWEEP_DONE ]; do
  if [ "$waited" -ge "$MAXWAIT" ]; then
    echo "ABORT: wave 1 produced no SWEEP_DONE after ${MAXWAIT}s; not launching wave 2"
    echo "aborted $(date -u +%FT%TZ)" > SWEEP_DONE_CANONICAL
    exit 1
  fi
  sleep 60
  waited=$((waited + 60))
done
echo "wave 1 finished ($(cat SWEEP_DONE)); starting wave 2 at $(date -u +%FT%TZ)"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="./const_lut_sac.py"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Identical to run_sweep.sh's COMMON except --anchor-policy.
COMMON="--addressing anchors --anchor-policy canonical_full_coverage \
        --forward-mode hard --nap 6 --tph 64 --heads 1 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

# Reduced to two arms on 2026-08-02, part-way through wave 1: `random` and `clumped` are
# dropped. `random` was always expected to be a near-null contrast (its constants sit
# half a bin width from `grid`'s -- see make_constants.py), and `clumped` only earns its
# place if `grid` beats `none`. The hypothesis lives or dies on none-vs-grid, so the
# budget goes to three clean seeds of those two.
ARMS="none grid"
SEEDS="0 1 2"
MAXJOBS=3

echo "XLA_FLAGS=$XLA_FLAGS"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

pids=()
n=0
for seed in $SEEDS; do
  for arm in $ARMS; do
    while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
      wait -n 2>/dev/null || true
      alive=()
      for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
      pids=("${alive[@]}")
    done
    n=$((n + 1))
    echo "=== launch $n/12  arm=$arm seed=$seed  canonical  $(date -u +%FT%TZ) ==="
    nohup $PY -u "$TRAIN" --seed "$seed" --constants "$arm" $COMMON \
          --tag "_c29c_${arm}_s${seed}" > "cellc_${arm}_s${seed}.log" 2>&1 &
    pids+=($!)
    sleep 25
  done
done

wait
echo "ALL 12 CANONICAL RUNS DONE $(date -u +%FT%TZ)"

for seed in $SEEDS; do
  for arm in $ARMS; do
    A="lut_sac_c29c_${arm}_s${seed}_actor.npz"
    [ -f "$A" ] || { echo "MISSING $A"; continue; }
    $PY -u eval_const_cpu.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
    $PY -u bit_usage.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
  done
done
echo "CANONICAL EVALS DONE $(date -u +%FT%TZ)"
date -u +%FT%TZ > SWEEP_DONE_CANONICAL
