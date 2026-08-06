#!/usr/bin/env bash
# exp_c42b — the two-venv parity test for LIFMultiHeadLUT, end to end.
#
# spiky/.venv has torch and no jax; walker2d_mjx/.venv has jax and no torch. The reference
# is dumped to an npz by one interpreter and asserted by the other.
#
# The torch module lives on branch exp/lif-detectors-mhl, extracted READ-ONLY into /tmp
# rather than vendored: a second copy in our tree can silently drift from the thing we
# claim parity with. Nothing on that branch is modified and nothing is checked out.
#
# TORCHDYNAMO_DISABLE=1: LIFMultiHeadLUT.forward carries an @torch.compile decorator. A
# compiled reference would be testing inductor's CPU lowering, not the module, and would
# make the parity numbers depend on the torch version's fusion choices. Eager is the
# reference.
set -eu
cd "$(dirname "$0")"

REF_BRANCH=24c0e60a   # PINNED: was a moving branch name; see note below
REF_SRC=src/spiky/lutorch/lif_multi_head_lut.py
STAGE=/tmp/mhl_ref_c42b
NPZ=$STAGE/torch_reference.npz

SPIKY_PY="$HOME/projects/spiky/.venv/bin/python"
MJX_PY="$HOME/projects/walker2d_mjx/.venv/bin/python"

mkdir -p "$STAGE/spiky/lutorch"
: > "$STAGE/spiky/__init__.py"
: > "$STAGE/spiky/lutorch/__init__.py"
git -C "$HOME/projects/spiky" show "$REF_BRANCH:$REF_SRC" \
    > "$STAGE/spiky/lutorch/lif_multi_head_lut.py"
python3 patch_torch_ref.py "$STAGE/spiky/lutorch/lif_multi_head_lut.py"
echo "staged $REF_BRANCH:$REF_SRC ($(git -C "$HOME/projects/spiky" rev-parse --short "$REF_BRANCH")) -> $STAGE  ($(wc -l < "$STAGE/spiky/lutorch/lif_multi_head_lut.py") lines)"

echo "--- torch reference (spiky venv, CPU, eager) ---"
TORCHDYNAMO_DISABLE=1 PYTHONPATH="$STAGE" "$SPIKY_PY" -u torch_ref_dump.py "$NPZ"

echo "--- JAX port (mjx venv, CPU) ---"
JAX_PLATFORMS=cpu "$MJX_PY" -u parity_check.py "$NPZ"
