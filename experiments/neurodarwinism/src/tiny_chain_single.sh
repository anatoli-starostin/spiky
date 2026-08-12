#!/bin/bash
# exp012: the single-output capacity test. The K=8 diagonal-LS symmetric config, but the task
# is ONE target dimension and the whole 48-neuron output budget reads it.
#
# N_TGT is unchanged (40 exc + 10 inh + 48 out = 98), so this is the same size of network and
# the same cost per round as run_diagls_k8 -- only the task and the output grouping differ.
#
# Which dimensions, and why these three. The brief asks for the best-correlated dimension and
# calls it dim 0, but the dissection's r=0.85 dimension is dim 5; dim 0 is r=0.43. So all
# three are run: dim 5 (the actual best), dim 1 (the actual worst, r=0.16), and dim 0 (the
# one the brief names), sequentially, GPU to itself.
#
# Per-dimension constant-predictor baselines on the held-out split -- the ONLY honest
# yardstick here, the 6-dim 34.152 is not:
#   dim 0  29.384    dim 1  24.535    dim 2  30.543
#   dim 3  42.147    dim 4  30.890    dim 5  47.412
set -u
SRC=/home/astarostin/projects/spiky/experiments/neurodarwinism/src
BASE=/home/astarostin/projects/spiky/experiments/neurodarwinism/exp012_tiny-direct-genome
PY=/home/astarostin/projects/spiky/.venv/bin/python
export TRITON_CACHE_DIR=/tmp/triton_cache MPLCONFIGDIR=/tmp/mpl
cd "$SRC" || exit 1

run_one () {
  DIM=$1; TAGP=$2; NOTE=$3
  echo "=== single-output run, target dim $DIM ($NOTE) ==="
  $PY -u tiny_run_full.py \
    --seeds 1 --pool 512 --cull 64 --rounds 1700 --batch 1024 --ckpt-every 50 \
    --crossover --runner tiny_grow_evolve.py --tag-prefix "$TAGP" \
    --task c6b5d8a5 \
    --extra "--lam 0 --mu 0 --p-grow 0 --p-shrink 0 --p-affine 0.25 --random-init \
--init-exc 40 --init-inh 10 --p-init 0.10 --quantized --weight-levels=-1.0,0,1.0 \
--delay-levels odd --fanout-cap 16 --inhibition-coeff-evolve --gain-evolve \
--max-episode-batch 128 --readout diagls --out-per-target 48 --out-agg mean \
--target-dims $DIM" \
    --out-dir "$BASE/run_single_t$DIM" \
    --label "exp012 SINGLE output, target dim $DIM ($NOTE)"
  echo "=== dim $DIM done ==="
}

run_one 5 S "best-correlated, r=0.85, own chance 47.41"
run_one 1 U "worst, r=0.16, own chance 24.54"
run_one 0 V "the dim the brief names, r=0.43, own chance 29.38"
echo "ALL THREE SINGLE-OUTPUT RUNS COMPLETE"
