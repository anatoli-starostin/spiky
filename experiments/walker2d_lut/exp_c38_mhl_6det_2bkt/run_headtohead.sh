#!/usr/bin/env bash
# exp_c38 — torch reference vs the JAX port, same config, same shapes, same GPU.
#
# Two venvs, so two processes: spiky/.venv has torch and no jax, walker2d_mjx/.venv has
# jax and no torch. The torch module is staged read-only out of git, exactly as
# run_parity.sh does it -- nothing on that branch is modified.
#
# RUN THIS ON AN IDLE GPU. Every number here is contended if a sweep is training, and a
# contended head-to-head is worthless because the two sides contend differently.
#
# The JAX side is swept over all three arrival-ordering spellings so the comparison also
# shows what that choice is worth. "lax_sort" is EXCLUDED under the determinism flag: its
# forward is fine but its VJP is a serialised scatter that does not finish (>400 s), which
# is the trap that cost a stalled sweep.
set -eu
cd "$(dirname "$0")"

REPS=${1:-50}
STAGE=/tmp/mhl_ref_c38
SPIKY_PY="$HOME/projects/spiky/.venv/bin/python"
MJX_PY="$HOME/projects/walker2d_mjx/.venv/bin/python"

mkdir -p "$STAGE/spiky/lutorch"
: > "$STAGE/spiky/__init__.py"
: > "$STAGE/spiky/lutorch/__init__.py"
git -C "$HOME/projects/spiky" show \
    origin/exp/lif-detectors-mhl:src/spiky/lutorch/lif_multi_head_lut.py \
    > "$STAGE/spiky/lutorch/lif_multi_head_lut.py"

echo "=== torch reference, COMPILED (as shipped) ==="
PYTHONPATH="$STAGE" "$SPIKY_PY" -u bench_torch_ref.py "$REPS"
echo
echo "=== torch reference, EAGER ==="
TORCHDYNAMO_DISABLE=1 PYTHONPATH="$STAGE" "$SPIKY_PY" -u bench_torch_ref.py "$REPS"
echo
echo "=== JAX port, WITH the determinism flag (the regime we train in) ==="
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_deterministic_ops=true CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$MJX_PY" -u bench_jax_actor.py "$REPS" rank,argsort
echo
echo "=== JAX port, WITHOUT the flag (lax_sort included) ==="
XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$MJX_PY" -u bench_jax_actor.py "$REPS" rank,argsort,lax_sort
