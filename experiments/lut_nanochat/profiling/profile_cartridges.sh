#!/bin/bash
# [lut_nanochat] Thin wrapper: per-op profile of every lutorch_ex cartridge (35 arms: pure, fused tier1/native/eager,
# fp32/bf16, Confidence/Quantised/Deploy) at the locked d24 geometry. All flags pass through to profile_cartridges.py
# (see --help). LUTORCH_EX_SRC=<repo>/src profiles another lutorch_ex checkout (e.g. main vs a PR branch).
# Same interpreter / TRITON_CACHE_DIR handling as profile_lut_ffn.sh.
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
        echo "[profile_cartridges.sh] $default not writable -> TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    fi
fi
echo "[profile_cartridges.sh] python: $PY"
exec "$PY" "$HERE/profile_cartridges.py" "$@"
