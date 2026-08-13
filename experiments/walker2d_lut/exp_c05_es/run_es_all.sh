#!/usr/bin/env bash
# Phase 3 + CMA-ES-on-LUT: the gradient-free ceiling for a small MLP, then the SAME
# loop pointed at a LUT policy via the bit-exact JAX forward (exp_c04).
# Sequential on purpose — the distillation sweep is using the same GPU.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--gens 150 --pop 128 --episodes 2 --horizon 400"

echo "=== MLP / OpenAI-ES  $(date -u +%FT%TZ) ==="
$PY -u es_mjx.py --policy mlp --algo openai $COMMON > es_mlp_openai.log 2>&1
echo "  rc=$?"

echo "=== LUT / OpenAI-ES  $(date -u +%FT%TZ) ==="
$PY -u es_mjx.py --policy lut --algo openai --nap 6 --tph 16 $COMMON > es_lut_openai.log 2>&1
echo "  rc=$?"

echo "=== MLP / sep-CMA-ES  $(date -u +%FT%TZ) ==="
$PY -u es_mjx.py --policy mlp --algo sepcma $COMMON > es_mlp_sepcma.log 2>&1
echo "  rc=$?"

echo "ALL ES RUNS DONE $(date -u +%FT%TZ)"
