#!/usr/bin/env bash
# Evaluation only — no training. Reads the smoke's headline number in context.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant
CK=/home/astarostin/projects/ckpt_backups/exp19_walker2d_lut/deploy_matched/actor_s2.pt
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$SRC"
export PYTHONPATH=/home/astarostin/projects/spiky/experiments/walker2d-lut/_qat_deps
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
$PY -u eval_qat_ckpt.py --ckpt "$CK" --envs 1024 --steps 2000 > "$OUT/eval_baseline.log" 2>&1
echo "rc=$?"
grep -v "load on device" "$OUT/eval_baseline.log" | tail -20
