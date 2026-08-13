#!/usr/bin/env bash
# exp_c13 — pre-draw every anchor cache SERIALLY before the sweep launches (#75).
#
# Why: the trainer generates its cache on a miss by shelling out to the spiky venv. With
# three trainers starting at once that is three concurrent writers, and although each
# (nap, tph, seed) maps to a DIFFERENT cache file today, relying on that is a race
# waiting to happen. Drawing them all up front is also a fail-fast check: if torch or
# lutorch is broken we find out in seconds, not 35 minutes into the first wave.
set -eu
cd "$(dirname "$0")"

PY="$HOME/projects/spiky/.venv/bin/python"
GEN="../exp_c11_lut_sac_2x2/gen_anchors.py"
POLICY=balanced

for seed in 0 1 2; do
  for nap in 6 7 8; do
    for tph in 32 64 128; do
      $PY "$GEN" --n-tables "$tph" --nap "$nap" --input-dim 17 --heads 1 \
                 --seed "$seed" --policy "$POLICY" --device cpu
    done
  done
done
echo "pre-generated 27 anchor caches (policy=$POLICY, device=cpu)"
