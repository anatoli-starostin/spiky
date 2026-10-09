#!/bin/bash
# [lut_nanochat] Thin wrapper: profile the d24 LUT-FFN block vs the dense d24 FFN on whatever CUDA GPU this is.
# All flags pass through to profile_lut_ffn.py (see its docstring / --help). Examples:
#   experiments/lut_nanochat/profiling/profile_lut_ffn.sh
#   experiments/lut_nanochat/profiling/profile_lut_ffn.sh --batch 16 --passes block profile --out-dir /tmp/x
# Interpreter: $PYTHON, else the nanochat uv venv if present, else python3 on PATH.
# Triton writes compiled kernels to TRITON_CACHE_DIR (default ~/.triton/cache); if that is unset AND the default is
# not writable (sandbox / read-only home) a per-user /tmp dir is used and printed - an explicit setting always wins.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x "$REPO/experiments/lut_nanochat/nanochat/.venv/bin/python" ]; then
        PY="$REPO/experiments/lut_nanochat/nanochat/.venv/bin/python"
    else
        PY="python3"
    fi
fi

if [ -z "${TRITON_CACHE_DIR:-}" ]; then
    default="${TRITON_HOME:-$HOME}/.triton/cache"
    if ! { mkdir -p "$default" 2>/dev/null && probe="$(mktemp -p "$default" 2>/dev/null)" && rm -f "$probe"; }; then
        export TRITON_CACHE_DIR="/tmp/triton-cache-$(id -u)"
        echo "[profile_lut_ffn.sh] $default not writable -> TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    fi
fi

echo "[profile_lut_ffn.sh] python: $PY"
exec "$PY" "$HERE/profile_lut_ffn.py" "$@"
