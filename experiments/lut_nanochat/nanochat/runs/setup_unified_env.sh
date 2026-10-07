#!/bin/bash
# [lut_nanochat] Establish the UNIFIED environment for the whole lut_nanochat line of work:
# the pinned nanochat uv venv (torch 2.9.1+cu128 from the vendored uv.lock) + spiky's lutorch_ex
# installed EDITABLE on top. This is for stages 2-3 (logits distillation, LUT-from-scratch); the
# dense d24 baseline does NOT import lutorch_ex, so the baseline is unaffected either way.
#
# Why this is safe for the baseline's citable pin:
#   lutorch_ex (src/spiky/lutorch_ex) declares ONLY `torch>=2.1` + `ninja` (no upper bound, no CUDA
#   pin), so the install is purely ADDITIVE: torch stays EXACTLY 2.9.1+cu128. We deliberately do NOT
#   add lutorch_ex to the vendored uv.lock/pyproject, so the baseline's pinned, diff-against-upstream
#   environment is untouched. lutorch_ex's CUDA kernels JIT-build at first use (needs nvcc + ninja;
#   verified building fine with the system CUDA 13.x toolkit against the cu128 torch).
#
# Run from experiments/lut_nanochat/nanochat, AFTER `uv sync --extra gpu` (RUNBOOK §4):
#   bash runs/setup_unified_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LX=../../../src/spiky/lutorch_ex                         # spiky repo root / src/spiky/lutorch_ex
BEFORE=$(.venv/bin/python -c "import torch; print(torch.__version__)")
uv pip install --python .venv/bin/python -e "$LX"
AFTER=$(.venv/bin/python -c "import torch; print(torch.__version__)")
if [ "$BEFORE" != "$AFTER" ]; then
    echo "ABORT: torch version changed ($BEFORE -> $AFTER) — the baseline pin must stay 2.9.1+cu128"; exit 1
fi
echo "torch unchanged: $AFTER"
NANOCHAT_FA3_REVISION=$(python3 -c "import json;print(json.load(open('../pins.json'))['fa3']['revision'])") \
    PYTHONPATH="$PWD" .venv/bin/python -c "import nanochat; import spiky.lutorch_ex; print('nanochat + spiky.lutorch_ex import OK')"
echo "UNIFIED ENV READY: torch $AFTER (pinned) + spiky.lutorch_ex (editable). Baseline unaffected."
