#!/usr/bin/env bash
# exp_c14 — the hyperplane x hard reference cell at THREE SEEDS (#75).
#
# exp_c13 put the anchors arm on 3 seeds and found a seed-sd up to 1850, but the 5146.9
# target it was measured against was still a SINGLE seed. This reseeds the reference so
# both sides of the headline comparison stand on the same footing.
#
# Config is exp_c11's hyperplane x hard verbatim -- nap6/tph32/heads1, ratio 0.5, 10,000
# iterations -- with only --seed varying. Hyperplane addressing LEARNS w and b, so it
# never touches the anchor sampler; --anchor-policy is irrelevant here and is not passed.
#
# All three run concurrently: ~1.1 GB each, and matching exp_c13's 3-way concurrency
# keeps the wall-clock per cell comparable (contention affects speed, not results).
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--addressing hyperplane --forward-mode hard --nap 6 --tph 32 --heads 1 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

for seed in 0 1 2; do
  echo "=== launch hyperplane/hard seed$seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$TRAIN" --seed "$seed" $COMMON \
        --tag "_c14_hyperplane_hard_s${seed}" > "cell_hyperplane_hard_s${seed}.log" 2>&1 &
  sleep 25   # stagger so the JIT compiles don't collide
done

wait
echo "ALL 3 CELLS DONE $(date -u +%FT%TZ)"
