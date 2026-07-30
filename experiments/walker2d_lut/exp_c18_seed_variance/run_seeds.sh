#!/usr/bin/env bash
# exp_c18 — CLEAN SEED-VARIANCE STUDY of hyperplane x hard, torch-faithful init (#75).
#
# WHY THIS RERUNS exp_c15 RATHER THAN REUSING IT. exp_c15 measured 3 seeds of this exact
# config and got 4318.5 +/- 178.0. exp_c16 then showed that at a FIXED seed the same
# command varies by ~663 return, so exp_c15's 178 was not a seed spread at all -- it was
# three draws from run noise that happened to land close. exp_c17 traced that noise to
# the atomics-based scatter-add in the table-weight backward and killed it with
# deterministic GPU ops (checkpoints came out bit-for-bit identical, 0/28,034 elements
# differing). Only now can "seed variance" be measured, because only now does a seed
# label a single reproducible run instead of a distribution.
#
# So: same config as exp_c15, 6 seeds instead of 3, determinism forced ON. exp_c15 is
# left intact; this is a new directory, not a rerun in place.
#
# COST OF DETERMINISM: ~+27% wall clock (exp_c17: 35.5 min/run vs ~28 without). Six runs
# 3-at-a-time is two waves, so expect roughly 1.5-2 h of training.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The exp_c17 fix, verbatim. Without BOTH of these a fixed seed does not reproduce.
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# --snap-every 500 records (w, b) and the row-update histogram 21 times per run. It reads
# state and consumes no randomness, so it cannot perturb the runs it is measuring; it
# exists so the "is the addressing still moving at 10k?" question can be answered if the
# spread turns out to be large. Cheaper to always record than to retrain to find out.
COMMON="--addressing hyperplane --hyperplane-init anchor_pairs \
        --hyperplane-anchor-policy canonical_full_coverage \
        --forward-mode hard --nap 6 --tph 32 --heads 1 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20 --snap-every 500"

SEEDS="0 1 2 3 4 5"
MAXJOBS=3            # ~7.5 GB each of 32 GB, as in exp_c13; 4-way crowds the desktop

echo "XLA_FLAGS=$XLA_FLAGS"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

pids=()
for seed in $SEEDS; do
  # Throttle by waiting on a PID (the exp_c13 form) -- `pgrep | wc -l` mis-parses.
  while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
    wait -n 2>/dev/null || true
    alive=()
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    pids=("${alive[@]}")
  done
  echo "=== launch seed $seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$TRAIN" --seed "$seed" $COMMON \
        --tag "_c18_seed${seed}" > "cell_seed${seed}.log" 2>&1 &
  pids+=($!)
  sleep 25   # stagger so the JIT compiles do not collide
done

wait
echo "ALL 6 SEEDS DONE $(date -u +%FT%TZ)"
