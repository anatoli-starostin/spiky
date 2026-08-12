#!/bin/bash
# exp012: the asymmetric-weight-grid arm of the 2x2, on the single-output task.
#
#   grid   excitatory {0.0, 0.1, .., 1.0}   11 levels, step 0.1
#          inhibitory {-1.0, 0.0}            2 levels, on/off; strength from inh_coeff
#
# Identical in every other respect to the {-1,0,1} single-output baselines in
# tiny_chain_single.sh -- same seed, pool, cull, rounds, batch, odd delays, fan-out cap 16,
# evolvable inh_coeff and gain, diagls readout, whole 48-neuron output budget on one target.
#
# p_weight is set EXPLICITLY to 0.25. The runner's auto-rule calls a grid binary when either
# Dale half has <= 2 levels, which the inhibitory half does, and would pick 0.5 -- but that
# rule is about grids where a weight carries one bit, and here the 11-level excitatory half
# holds 481 of the 539 initial synapses.
#
# Waits for the {-1,0,1} chain to finish so the two never share the GPU.
set -u
SRC=/home/astarostin/projects/spiky/experiments/neurodarwinism/src
BASE=/home/astarostin/projects/spiky/experiments/neurodarwinism/exp012_tiny-direct-genome
PY=/home/astarostin/projects/spiky/.venv/bin/python
export TRITON_CACHE_DIR=/tmp/triton_cache MPLCONFIGDIR=/tmp/mpl
cd "$SRC" || exit 1

echo "waiting for the {-1,0,1} single-output chain to finish..."
while pgrep -f "tiny_chain_single.sh" > /dev/null; do sleep 60; done
while pgrep -f "tiny_grow_evolve.py" > /dev/null; do sleep 60; done
echo "GPU free; starting the asymmetric-grid runs"

LEVELS=-1.0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

run_one () {
  DIM=$1; TAGP=$2; NOTE=$3
  echo "=== asym-grid single-output run, target dim $DIM ($NOTE) ==="
  $PY -u tiny_run_full.py \
    --seeds 1 --pool 512 --cull 64 --rounds 1700 --batch 1024 --ckpt-every 50 \
    --crossover --runner tiny_grow_evolve.py --tag-prefix "$TAGP" \
    --task 61cec856 \
    --extra "--lam 0 --mu 0 --p-grow 0 --p-shrink 0 --p-affine 0.25 --random-init \
--init-exc 40 --init-inh 10 --p-init 0.10 --quantized --weight-levels=$LEVELS \
--p-weight 0.25 --delay-levels odd --fanout-cap 16 --inhibition-coeff-evolve --gain-evolve \
--max-episode-batch 128 --readout diagls --out-per-target 48 --out-agg mean \
--target-dims $DIM" \
    --out-dir "$BASE/run_single_asym_t$DIM" \
    --label "exp012 SINGLE output, ASYM grid (11 exc / 2 inh), target dim $DIM ($NOTE)"
  echo "=== asym dim $DIM done ==="
}

run_one 0 A "own chance 29.38"
run_one 1 B "own chance 24.54"
echo "BOTH ASYMMETRIC-GRID RUNS COMPLETE"
