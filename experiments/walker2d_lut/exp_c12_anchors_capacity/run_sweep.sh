#!/usr/bin/env bash
# exp_c12 — nap x tph capacity sweep for the anchors x hard LUT-SAC cell (#75).
#
# Question: can added capacity close fixed-anchor addressing's gap to learned
# hyperplanes (anchors x hard 4302.4 vs hyperplane x hard 5146.9, both at 28k params)?
#
# Everything fixed except nap and tph: anchors (frozen), hard forward, ratio 0.5,
# 10,000 iterations, same env and optimizer as the exp_c11 2x2. Baseline nap6/tph32
# is NOT rerun -- its 4302.4 +/- 49.9 is reused.
#
# Ordered so the most informative cells land first (diagonal + high-capacity corner),
# in case the sweep is cut short. Three concurrent, matching the measured headroom.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--addressing anchors --forward-mode hard --iters 10000 --envs 64 \
        --rollout 1 --updates 32 --batch 512 --warmup 500 --row-clip 1.0 \
        --eval-every 500 --eval-episodes 20"

CELLS="7:64 8:128 6:128 8:32 6:64 7:32 7:128 8:64"
MAXJOBS=3

for cell in $CELLS; do
  nap="${cell%%:*}"; tph="${cell##*:}"
  # throttle to MAXJOBS concurrent trainers
  while [ "$(pgrep -f "lut_sac.py .*--tag _c12_" | wc -l)" -ge "$MAXJOBS" ]; do
    sleep 20
  done
  echo "=== launch nap$nap tph$tph  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$TRAIN" --nap "$nap" --tph "$tph" $COMMON \
        --tag "_c12_nap${nap}_tph${tph}" > "cell_nap${nap}_tph${tph}.log" 2>&1 &
  sleep 25   # stagger so the JIT compiles don't collide
done

while pgrep -f 'lut_sac.py .*--tag _c12_' > /dev/null; do sleep 30; done
echo "ALL CELLS DONE $(date -u +%FT%TZ)"
