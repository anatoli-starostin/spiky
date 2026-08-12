#!/bin/bash
# Re-run the three headline arms with the fixed --readout linear, each from the SAME start it
# used before so the A/B is clean. Sequential, because each is pool 512 / batch 1024 and the
# engine's batch x neurons x ticks ceiling tightens when two of them share the GPU.
set -u
SRC=/home/astarostin/projects/spiky/experiments/neurodarwinism/src
BASE=/home/astarostin/projects/spiky/experiments/neurodarwinism/exp012_tiny-direct-genome
PY=/home/astarostin/projects/spiky/.venv/bin/python
export TRITON_CACHE_DIR=/tmp/triton_cache MPLCONFIGDIR=/tmp/mpl
cd "$SRC" || exit 1

echo "waiting for the production run and the A/B probes to finish..."
until [ -f "$BASE/full_run_q3sym_coeff_random/y0_final.json" ]; do sleep 30; done
while pgrep -f "tiny_grow_evolve.py.*--tag ab_" > /dev/null; do sleep 30; done
echo "GPU free; starting the re-runs"

COMMON="--seeds 1 --pool 512 --cull 64 --rounds 1700 --batch 1024 --ckpt-every 50 \
--crossover --runner tiny_grow_evolve.py"

# ---- 1. fixed 40/10, continuous weights (the full_run_fixed_40_10_random config)
echo "[1/3] fixed 40/10 continuous"
$PY -u tiny_run_full.py $COMMON --tag-prefix L \
  --extra "--lam 0 --mu 0 --p-grow 0 --p-shrink 0 --p-affine 0.25 --random-init \
--init-exc 40 --init-inh 10 --p-init 0.10 --max-episode-batch 256 --readout linear" \
  --out-dir "$BASE/rerun_fixed_40_10_linear" \
  --label "RERUN fixed 40/10 continuous, LINEAR readout" > /tmp/rerun1.log 2>&1
echo "[1/3] done"

# ---- 2. growable (the full_run_grow_calib config: lambda 0.05, grow/shrink on, SEEDED)
echo "[2/3] growable"
$PY -u tiny_run_full.py $COMMON --tag-prefix M \
  --extra "--lam 0.05 --p-affine 0.25 --max-episode-batch 256 --readout linear" \
  --out-dir "$BASE/rerun_grow_calib_linear" \
  --label "RERUN growable lambda 0.05, LINEAR readout" > /tmp/rerun2.log 2>&1
echo "[2/3] done"

# ---- 3. quantized symmetric grid + inh_coeff + gain + fanout16 + odd delays
echo "[3/3] quantized symmetric"
$PY -u tiny_run_full.py $COMMON --tag-prefix N \
  --extra "--lam 0 --mu 0 --p-grow 0 --p-shrink 0 --p-affine 0.25 --random-init \
--init-exc 40 --init-inh 10 --p-init 0.10 --quantized --weight-levels=-1.0,0,1.0 \
--delay-levels odd --fanout-cap 16 --inhibition-coeff-evolve --gain-evolve \
--max-episode-batch 256 --readout linear" \
  --out-dir "$BASE/rerun_q3sym_linear" \
  --label "RERUN quantized symmetric + coeff + gain, LINEAR readout" > /tmp/rerun3.log 2>&1
echo "[3/3] done -- all three re-runs complete"
