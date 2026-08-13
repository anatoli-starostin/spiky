#!/usr/bin/env bash
# exp_c21 — seed 4 at DOUBLE the budget: 20,000 iters instead of 10,000 (#75).
#
# Motivation from exp_c18's diagnostics: the addressing had NOT converged at 10k. Late
# movement was 0.25x early, but 2.5-3.2% of address bits were still being rewritten in the
# final 2,000 iterations across every seed. 10k was a cut-off, not a resting point.
#
# Seed 4 is the one to extend: at 10k it reached 5286.6 by finding a 4.29 m/s gait, and
# exp_c20 showed that solution is carried by its addressing -- which is exactly the part
# still in motion.
#
# EVERY OTHER KNOB IS exp_c18's, VERBATIM, including seed=4, so this is a controlled
# extension of one run rather than a new configuration. In particular the anchor draw is
# unchanged: it comes from lutorch's torch generator keyed by the integer seed, not from the
# jax stream, so seed=4 reproduces the identical starting addressing.
#
# --checkpoint-at 10000 keeps an extra, never-overwritten checkpoint at the 10k mark. Under
# determinism that state should be BIT-IDENTICAL to exp_c18 seed 4's final checkpoint --
# `iters` only controls the loop length, and nothing downstream of it feeds the RNG chain.
# collect.py asserts that identity rather than assuming it: if it holds, the 10k->20k gain
# is measured on genuinely the same trajectory, and 5286.6 is the 10k number for free.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

SLACK_TASK="${SLACK_TASK:-b772590a}"
nohup setsid "$PY" -u slack_bar.py --task "$SLACK_TASK" --interval 150 \
      > slack_bar.log 2>&1 &
BAR_PID=$!

echo "XLA_FLAGS=$XLA_FLAGS"
echo "=== seed 4, 20,000 iters (alone on the GPU)  $(date -u +%FT%TZ) ==="

# Sole occupant of the GPU, so this runs in the foreground -- no PID array to wait on and
# therefore none of the bare-`wait` deadlock that bit exp_c19.
$PY -u "$TRAIN" \
    --addressing hyperplane --hyperplane-init anchor_pairs \
    --hyperplane-anchor-policy canonical_full_coverage \
    --forward-mode hard --nap 6 --tph 32 --heads 1 --seed 4 \
    --iters 20000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
    --row-clip 1.0 --eval-every 500 --eval-episodes 20 \
    --snap-every 500 --checkpoint-at 10000 \
    --tag "_c21_seed4_20k" > cell_seed4_20k.log 2>&1
echo "  rc=$?  $(date -u +%FT%TZ)"

echo "TRAINING DONE $(date -u +%FT%TZ) — evaluating"
$PY -u collect.py
echo "LONG RUN EVAL DONE $(date -u +%FT%TZ)"
