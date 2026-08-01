#!/usr/bin/env bash
# exp_c25 — four arms, launched together on the one GPU.
#
# Concurrency is measured, not hoped for: solo 600 iters takes 183 s, three
# concurrent take 211/215/215 s (1.17x each, spread under 2%), because the trainer
# syncs to host 32 times per iteration for the coverage histogram and therefore
# leaves the GPU idle at a median 5% utilisation. Peak memory for three arms was
# 3.9 GB of 32.6 GB. Running all four in the SAME window also keeps machine
# conditions identical across arms, which staggering them would not.
#
# Everything except --phase-freq is c21's configuration at seed 4.
set -u
cd "$(dirname "$0")"
PY=/home/astarostin/projects/walker2d_mjx/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ITERS=10000
COMMON="--addressing hyperplane --hyperplane-init anchor_pairs
        --hyperplane-anchor-policy canonical_full_coverage --forward-mode hard
        --nap 6 --tph 32 --heads 1 --seed 4
        --iters $ITERS --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

echo "=== exp_c25 sweep, $ITERS iters x 4 arms, launched $(date -u +%FT%TZ) ==="
for F in 0 0.85 1.703 2.55; do
  TAG="_c25_f${F//./p}"
  $PY -u phase_lut_sac.py $COMMON --phase-freq "$F" --tag "$TAG" \
      > "run$TAG.log" 2>&1 &
  echo "  launched f=$F  pid $!  -> run$TAG.log"
done
wait
echo "=== all arms finished $(date -u +%FT%TZ) ==="

for F in 0 0.85 1.703 2.55; do
  TAG="_c25_f${F//./p}"
  echo "--- CPU reference, f=$F ---"
  $PY -u eval_phase_cpu.py "lut_sac${TAG}_actor.npz" --episodes 100 \
      2>&1 | grep -v "Failed to import"
done
echo "=== sweep + eval done $(date -u +%FT%TZ) ==="
