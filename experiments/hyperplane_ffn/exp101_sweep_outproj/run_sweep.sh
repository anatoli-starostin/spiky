#!/usr/bin/env bash
# exp101 out_proj sweep orchestrator: 12 runs (4k steps each), 3 concurrent, 4 waves.
# Grid: out_proj inner_in IN in {24,48,96} x tph TPH in {16,32,64,128}. q/k/v held at
# exp101 baseline. Each run writes to a per-run subdir in{IN}_tph{TPH}/ (no collisions).
# Run via: sbox bash run_sweep.sh   (one cage process; internal &/wait stay in-cage)
set -u
D=/home/astarostin/projects/spiky/experiments/hyperplane_ffn/exp101_sweep_outproj
cd "$D"
mkdir -p sweeplogs
PY=../../../.venv/bin/python
combos=("24 16" "24 32" "24 64" "24 128" "48 16" "48 32" "48 64" "48 128" "96 16" "96 32" "96 64" "96 128")

run_one() {  # IN TPH
  local IN=$1 TPH=$2 tag="in${1}_tph${2}"
  OUT_IN=$IN OUT_TPH=$TPH RUN_TAG=$tag N_STEPS=4000 \
    TRITON_CACHE_DIR="/tmp/triton_$tag" MPLCONFIGDIR=/tmp/mplconfig \
    "$PY" -u train_sweep.py > "sweeplogs/$tag.log" 2>&1
}

i=0; wave=1; n=${#combos[@]}
while [ $i -lt $n ]; do
  echo "=== WAVE $wave launching: ${combos[$i]:-} | ${combos[$((i+1))]:-} | ${combos[$((i+2))]:-} ==="
  pids=()
  for j in 0 1 2; do
    idx=$((i+j)); [ $idx -ge $n ] && break
    set -- ${combos[$idx]}
    run_one "$1" "$2" & pids+=("$!")
  done
  wait "${pids[@]}"
  echo "=== WAVE $wave complete ==="
  i=$((i+3)); wave=$((wave+1))
done
echo "=== SWEEP COMPLETE (12/12) ==="
