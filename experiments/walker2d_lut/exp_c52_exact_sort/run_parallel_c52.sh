#!/usr/bin/env bash
# exp_c52 — the exact-sort ablation: exp_c50 with SORT_FORM flipped rank -> argsort.
#
# Everything else is exp_c50 verbatim: current unified LIFMultiHeadLUT, 1 head x 128 tables
# x 1 detector x 16 buckets, per-table betas, stock 0.1 table init, delay_init_std=0,
# freeze_temperature=False, and the delay clamp with its LOWER bound removed
# (`clamp(delay, -inf, t_window)`, upper cap kept). Seeds 0/1/2, the same three c36, c48,
# c49 and c50 ran.
#
# WHAT IT ISOLATES. c36 ordered arrivals with `jnp.argsort(a, axis=-1, stable=True)` +
# `take_along_axis` (jax_bucket_lif.py:207). c50 uses the sort-free `rank` form:
# rank_k = #{j : a_j < a_k}, ties by index, applied as a one-hot contraction. This run puts
# c36's exact spelling back, so rank-vs-sort is tested as a candidate for the residual
# c50-vs-c36 gap.
#
# WHAT `sort_equivalence.py` ALREADY SHOWS, and read this before spending the GPU time.
# The two forms are BIT-IDENTICAL: every intermediate, both forwards and the gradient of
# all 8 parameters agree to exactly 0.000e+00, on a fresh init, on perturbed random
# weights, on all three trained c50 checkpoints, AND on a constructed case where 100% of
# adjacent arrival pairs are exactly tied so the tie-break alone decides the permutation.
# Since the gradients are identical, the training trajectory is identical, so this sweep is
# expected to reproduce c50's seeds 0/1/2 EXACTLY -- 4447.2 / 3719.6 / 1156.3.
#
# It is run anyway because it was asked for and because a bit-identity argument is a claim
# about the code while a matching sweep is a measurement of the system. But its cost is
# real: c36's argsort spelling took 240.5 min per seed against c50's 37.4, because
# `jnp.argsort` alone costs 19-22 ms and its `take_along_axis` VJP is a scatter-add that
# `--xla_gpu_deterministic_ops=true` serialises. Expect ~4 hours, not ~40 minutes.
#
# Config comes from the trainer defaults (heads=1, tph=128, ndet=1, buckets=16,
# delay_init_std=0, delay_init_const=0, table_init_std=0.1 STOCK, share_betas=0,
# freeze_temperature=0); passing them as flags would only invite a transcription error in
# the least visible place. The ONE thing not defaulted is SORT_FORM, which is a module
# constant in jax_mhl_lut.py -- set to "argsort" in this directory's copy.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local S=$1 TAG="_c52_s$1"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

for S in 0 1 2; do cell "$S" & done
wait

touch SWEEP_DONE_C52
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
