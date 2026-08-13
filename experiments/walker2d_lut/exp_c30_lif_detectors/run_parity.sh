#!/usr/bin/env bash
# exp_c30 — the two-venv parity test, end to end.
#
# The torch reference and the JAX port cannot meet in one process: spiky/.venv has torch
# and no jax, walker2d_mjx/.venv has jax and no torch. So the reference is dumped to an
# npz by one interpreter and asserted by the other.
#
# The torch module lives on branch exp/lif-detectors-mhl, which is NOT the branch this
# experiment sits on. It is extracted read-only into /tmp rather than vendored into our
# tree: copying another branch's source in would create a second copy that can silently
# drift from the thing we claim parity with.
set -eu
cd "$(dirname "$0")"

REF_BRANCH=912d3f99   # PINNED: was a moving branch name; see note below
REF_SRC=src/spiky/lutorch/lif_detectors_mhl.py
STAGE=/tmp/lif_ref
NPZ=/tmp/lif_ref/torch_reference.npz

SPIKY_PY="$HOME/projects/spiky/.venv/bin/python"
MJX_PY="$HOME/projects/walker2d_mjx/.venv/bin/python"

mkdir -p "$STAGE/spiky/lutorch"
: > "$STAGE/spiky/__init__.py"
: > "$STAGE/spiky/lutorch/__init__.py"
git -C "$HOME/projects/spiky" show "$REF_BRANCH:$REF_SRC" > "$STAGE/spiky/lutorch/lif_detectors_mhl.py"
echo "staged $REF_BRANCH:$REF_SRC -> $STAGE  ($(wc -l < "$STAGE/spiky/lutorch/lif_detectors_mhl.py") lines)"

echo "--- torch reference (spiky venv, CPU) ---"
PYTHONPATH="$STAGE" "$SPIKY_PY" -u torch_ref_dump.py "$NPZ"

echo "--- JAX port (mjx venv) ---"
JAX_PLATFORMS=cpu "$MJX_PY" -u parity_check.py "$NPZ"
