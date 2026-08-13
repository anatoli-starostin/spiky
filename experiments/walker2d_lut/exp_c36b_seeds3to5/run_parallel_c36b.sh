#!/usr/bin/env bash
# exp_c36b — three MORE seeds (3, 4, 5) of exp_c36, co-resident on the 5090.
#
# NOT a new configuration and NOT a port. This is exp_c36 continued: the SAME
# `bucket_sac.py` and the SAME `jax_bucket_lif.py` the original three seeds ran, copied
# unmodified from exp_c36_bucket_tables, on the old pre-refactor BucketLIFDetectorsMHL
# architecture -- 1 head x 128 tables x 16 buckets, no delay clamp at all, trainable
# temperatures.
#
# WHY. exp_c36 is the anchor the entire c48->c53 bisect is measured against: 4246.1 +/-
# 298.4 with 3/3 takeoff. Everything since has been read as a shortfall against it. But
# c36 is n=3, and the configuration lineage it belongs to is bimodal -- exp_c50 pooled to
# n=9 measures a takeoff rate of 4/9 = 0.444, under which a 3/3 result has probability
# 0.444^3 ~= 8.8%. Unlikely, not negligible. If c36 drew a lucky three then part of the
# "residual gap" the bisect is chasing does not exist.
#
# So this run tests the anchor itself. Pooled to n=6 it either holds near 4246 -- and the
# gap is real and the bisect continues -- or it falls toward c50's 2700 and the whole
# c48-c53 shortfall narrative needs revising.
#
# THE CONFIG IS c36's, taken from its own run JSONs rather than retyped: heads 1, tph 128,
# buckets 16, iters 10000, envs 64, rollout 1, updates 32, batch 512, buffer 1e6, warmup
# 500, lr 3e-4, actor_lr 3e-4, gamma 0.99, tau 0.005, target_entropy -6.0, row_clip 1.0,
# eval_every 500, eval_episodes 20. Those are exactly `bucket_sac.py`'s defaults, which is
# why the original c36 passed only --seed and --tag, and why this does too: retyping them
# as flags would only invite a transcription error in the least visible place.
#
# COST. c36 measured 240.5 min/seed with 3 co-resident, against 37.4 for the MHL runs --
# the old module's O(N) cumsum membrane and exact-argsort ordering are simply slower.
# Expect ~4 hours, not ~40 minutes.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local S=$1 TAG="_c36_s$1"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u bucket_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_bucket_cpu.py "bucket_sac${TAG}_actor.npz" --episodes 100 \
      >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

for S in 3 4 5; do cell "$S" & done
wait

touch SWEEP_DONE_C36B
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
