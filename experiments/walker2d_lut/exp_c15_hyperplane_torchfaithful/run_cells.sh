#!/usr/bin/env bash
# exp_c15 — hyperplane x hard at 3 seeds with TORCH'S DEFAULT INIT (#75).
#
# exp_c14 ran this config with the legacy JAX init (dense w ~ N(0, 0.5^2), b ~ N(0, 0.1^2))
# and got 3968.3 +/- 1186.6. This reruns it with hyperplane_init="anchor_pairs" -- w =
# e_a - e_b, b = 0, drawn by lutorch's own CANONICAL_FULL_COVERAGE sampler -- so the model
# STARTS as a bit-exact FastMultiHeadLut and learns away, exactly as torch does by default.
# Verified bit-exact against torch at init by exp_c11/verify_hp_init.py.
#
# Everything else is exp_c14 verbatim, so the two are a controlled A/B on the init alone.
# exp_c14 is left intact.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--addressing hyperplane --hyperplane-init anchor_pairs \
        --hyperplane-anchor-policy canonical_full_coverage \
        --forward-mode hard --nap 6 --tph 32 --heads 1 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

for seed in 0 1 2; do
  echo "=== launch hyperplane/hard/anchor_pairs seed$seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$TRAIN" --seed "$seed" $COMMON \
        --tag "_c15_hp_anchorinit_s${seed}" > "cell_hp_anchorinit_s${seed}.log" 2>&1 &
  sleep 25
done

wait
echo "ALL 3 CELLS DONE $(date -u +%FT%TZ)"
