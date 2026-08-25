#!/usr/bin/env bash
# exp_g_0017 q/k/v ATTENTION sweep orchestrator: 9 runs (4k steps each), 2 concurrent, 5 waves.
#
# Grid: q/k/v inner_in QKV_IN in {24,48,96} x tph QKV_TPH in {32,48,64}.
# out_proj is PINNED at inner_in=48 / tph=64 / n_heads=8 / inner_out=48 / nap=6.
# Each run writes to a per-run subdir qkvin{IN}_tph{TPH}/ (no collisions).
#
# Run via: sbox bash run_sweep.sh   (one cage process; internal &/wait stay in-cage)
#
# TWO DELIBERATE DIFFERENCES FROM nebius's exp101_sweep_outproj/run_sweep.sh:
#
#  1. 2-WIDE BY EXPLICIT INSTRUCTION. Measured peak GPU on this 5090 (32,607 MiB
#     total) for the largest point of THIS grid (96/64): 10,304 MiB, flat across
#     40 one-second samples (the caching allocator holds a steady reservation, so
#     there are no transient spikes to absorb).
#       3 x 10,304 = 30,912 MiB -> WOULD fit, but only ~1.7 GB spare (5%)
#       2 x 10,304 = 20,608 MiB -> ~12.0 GB spare (37%)
#     3-wide is arithmetically viable; Anatoly chose 2-wide anyway. WIDE=3 is one
#     env var away if that is ever revisited.
#     (For the record, the earlier 12-point grid reached 14,240 MiB at 96/128,
#     where 3-wide genuinely did NOT fit.)
#
#  2. ABSOLUTE PATHS. nebius's script hardcodes ~/projects/spiky and a relative
#     ../../../.venv python. On gpustar the branch lives in the WORKTREE
#     ~/projects/spiky-fmhl-next (the primary checkout is on live/walker2d-viz and
#     has no .venv sibling at that relative depth), so both are absolute here.
set -u
D=/home/astarostin/projects/spiky-fmhl-next/experiments/hyperplane_ffn/exp_g_0017_sweep_qkv
cd "$D"
mkdir -p sweeplogs
PY=/home/astarostin/projects/spiky/.venv/bin/python
SRC=/home/astarostin/projects/spiky-fmhl-next/src

combos=("24 32" "24 48" "24 64" "48 32" "48 48" "48 64" "96 32" "96 48" "96 64")
WIDE=${WIDE:-2}

run_one() {  # IN TPH
  local IN=$1 TPH=$2 tag="qkvin${1}_tph${2}"
  QKV_IN=$IN QKV_TPH=$TPH RUN_TAG=$tag N_STEPS=${N_STEPS:-4000} \
    TRITON_CACHE_DIR="/tmp/triton_$tag" MPLCONFIGDIR=/tmp/mplconfig \
    NANOCHAT_ROOT=/home/astarostin/projects/nanochat PYTHONPATH="$SRC" \
    "$PY" -u train_sweep.py > "sweeplogs/$tag.log" 2>&1
}

i=0; wave=1; n=${#combos[@]}
while [ $i -lt $n ]; do
  msg=""
  for j in $(seq 0 $((WIDE-1))); do
    idx=$((i+j)); [ $idx -ge $n ] && break
    msg="$msg | ${combos[$idx]}"
  done
  echo "=== WAVE $wave launching:$msg ==="
  pids=()
  for j in $(seq 0 $((WIDE-1))); do
    idx=$((i+j)); [ $idx -ge $n ] && break
    set -- ${combos[$idx]}
    run_one "$1" "$2" & pids+=("$!")
  done
  wait "${pids[@]}"
  echo "=== WAVE $wave complete ==="
  i=$((i+WIDE)); wave=$((wave+1))
done
echo "=== SWEEP COMPLETE (${n}/${n}) ==="
