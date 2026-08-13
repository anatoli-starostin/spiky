#!/usr/bin/env bash
# exp_c27 — three QAT arms (K = 7, 5, 3), concurrent, then the CPU eval on each grid.
# c21's configuration at seed 4, 10k iters; only --action-levels differs.
# exp_c25 measured this trainer at 1.17x per arm for 3-way concurrency with under 2%
# spread between arms; GPU was verified idle (407 MiB / 0%) before launch.
set -u
cd "$(dirname "$0")"
PY=/home/astarostin/projects/walker2d_mjx/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--addressing hyperplane --hyperplane-init anchor_pairs
        --hyperplane-anchor-policy canonical_full_coverage --forward-mode hard
        --nap 6 --tph 32 --heads 1 --seed 4
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

echo "=== exp_c27 QAT sweep, K in {7,5,3}, launched $(date -u +%FT%TZ) ==="
for K in 7 5 3; do
  $PY -u qat_lut_sac.py $COMMON --action-levels "$K" --tag "_c27_K$K" \
      > "run_c27_K$K.log" 2>&1 &
  echo "  launched K=$K  pid $!"
done
wait
echo "=== training done $(date -u +%FT%TZ) ==="

for K in 7 5 3; do
  $PY -u eval_qat_cpu.py "lut_sac_c27_K${K}_actor.npz" --episodes 100 \
      2>&1 | grep -v "Failed to import"
done
echo "=== qat sweep + eval done $(date -u +%FT%TZ) ==="
