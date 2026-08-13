#!/usr/bin/env bash
# exp_c32 — the two-venv parity test for BucketLIFDetectorsMHL, end to end.
#
# spiky/.venv has torch and no jax; walker2d_mjx/.venv has jax and no torch. The reference
# is dumped to an npz by one interpreter and asserted by the other.
#
# The torch module lives on branch exp/lif-detectors-mhl, extracted READ-ONLY into /tmp
# rather than vendored: a second copy in our tree can silently drift from the thing we
# claim parity with. Nothing on that branch is modified.
set -eu
cd "$(dirname "$0")"

REF_BRANCH=0024b81f   # PINNED: was a moving branch name; see note below
REF_SRC=src/spiky/lutorch/bucket_lif_detectors_mhl.py
STAGE=/tmp/bucket_lif_ref
NPZ=/tmp/bucket_lif_ref/torch_reference.npz

SPIKY_PY="$HOME/projects/spiky/.venv/bin/python"
MJX_PY="$HOME/projects/walker2d_mjx/.venv/bin/python"

mkdir -p "$STAGE/spiky/lutorch"
: > "$STAGE/spiky/__init__.py"
: > "$STAGE/spiky/lutorch/__init__.py"
git -C "$HOME/projects/spiky" show "$REF_BRANCH:$REF_SRC" \
    > "$STAGE/spiky/lutorch/bucket_lif_detectors_mhl.py"
echo "staged $REF_BRANCH:$REF_SRC -> $STAGE  ($(wc -l < "$STAGE/spiky/lutorch/bucket_lif_detectors_mhl.py") lines)"

echo "--- torch reference (spiky venv, CPU) ---"
PYTHONPATH="$STAGE" "$SPIKY_PY" -u torch_ref_dump.py "$NPZ"

echo "--- JAX port (mjx venv) ---"
JAX_PLATFORMS=cpu "$MJX_PY" -u parity_check.py "$NPZ"
