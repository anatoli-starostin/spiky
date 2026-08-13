#!/usr/bin/env bash
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant/qat_n22_full/probe
PAR=/home/astarostin/projects/ckpt_backups/exp19_walker2d_lut/deploy_matched/actor_s2.pt
FT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant/qat_n22_full
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$SRC"
export PYTHONPATH=/home/astarostin/projects/spiky/experiments/walker2d-lut/_qat_deps
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "$OUT"

$PY -u probe_raw_readout.py --ckpt "$PAR" --label "parent (deploy_matched s2)" \
    --out "$OUT/parent.json" 2>&1 | grep -v "load on device"
for s in 0 1 2; do
  $PY -u probe_raw_readout.py --ckpt "$FT/qat_s${s}.pt" --label "QAT n22 seed $s" \
      --out "$OUT/qat_s${s}.json" 2>&1 | grep -v "load on device"
done
echo PROBE_DONE
