#!/bin/bash
# exp012: the radically simplified single-output net.
#
#   17 inputs -> 8 excitatory hidden -> 1 output.  No inhibition at all.
#   26 neurons per candidate, against the 98 of the run_single_* generation.
#
# Weight grid is the excitatory half only, {0.0, 0.1, .., 1.0}; there is no negative Dale
# half to have, so inh_coeff is dropped (the runner refuses the combination). Gain is still
# evolved. Everything else matches the recent single-output setup: odd delays, TTFS
# first-spike readout, diagls scale+shift fitted on the training batch and carried unchanged
# to held-out, 96-tick episodes, MSE against the centred/quantised-to-32 target.
#
# --fanout-cap is deliberately absent: an excitatory row can reach at most 8 exc + 1 out = 9
# cells, so any cap of 16 would be decoration.
#
# --max-episode-batch 512: at 26 neurons x 512 candidates the engine's batch x neurons x
# ticks ceiling (~1.4e9) allows ~1096, so 512 is two chunks with a wide margin -- four times
# fewer chunks than the 98-neuron runs needed.
#
# Per-dimension constant-predictor baselines (the only honest yardstick here):
#   dim 0  29.384   dim 1  24.535   dim 2  30.543   dim 3  42.147   dim 4  30.890   dim 5  47.412
set -u
SRC=/home/astarostin/projects/spiky/experiments/neurodarwinism/src
BASE=/home/astarostin/projects/spiky/experiments/neurodarwinism/exp012_tiny-direct-genome
PY=/home/astarostin/projects/spiky/.venv/bin/python
export TRITON_CACHE_DIR=/tmp/triton_cache MPLCONFIGDIR=/tmp/mpl
cd "$SRC" || exit 1

LEVELS=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

run_one () {
  DIM=$1; TAGP=$2; NOTE=$3
  echo "=== 8-exc single-output run, target dim $DIM ($NOTE) ==="
  $PY -u tiny_run_full.py \
    --seeds 1 --pool 512 --cull 64 --rounds 1700 --batch 1024 --ckpt-every 50 \
    --crossover --runner tiny_grow_evolve.py --tag-prefix "$TAGP" \
    --task 468768cb \
    --extra "--lam 0 --mu 0 --p-grow 0 --p-shrink 0 --p-affine 0.25 --random-init \
--hidden-capacity 8,0 --init-exc 8 --init-inh 0 --p-init 0.5 --quantized \
--weight-levels=$LEVELS --p-weight 0.25 --delay-levels odd --gain-evolve \
--max-episode-batch 512 --readout diagls --out-per-target 1 --out-agg mean \
--target-dims $DIM" \
    --out-dir "$BASE/run_tiny8_t$DIM" \
    --label "exp012 TINY 17-8-1, no inhibition, target dim $DIM ($NOTE)"
  echo "=== tiny8 dim $DIM done ==="
}

run_one 0 T "own chance 29.38 -- the brief's default"
run_one 5 W "own chance 47.41 -- the only dim with a 50-neuron reference (12.49)"
echo "BOTH TINY RUNS COMPLETE"
