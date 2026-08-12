#!/bin/bash
# exp012 A/B: the K=8 diagonal-LS symmetric config, changing ONLY the delay grid.
#
#   run_diagls_k8   delay grid = odd ticks 1,3,..,63   (32 levels, mutation hop = 2 ticks)
#   this run        delay grid = every tick 1..64      (64 levels, mutation hop = 1 tick)
#
# Everything else is byte-identical to the config recorded in run_diagls_k8/P0_final.json:
# seed 0, pool 512, cull 64, 1700 rounds, batch 1024, sigma 0.045, p_delay 0.08, p_weight 0.5,
# crossover, quantized {-1,0,+1}, fanout cap 16, gain + inh_coeff evolved, readout diagls,
# 8 output neurons per target aggregated by mean, random init 40 exc / 10 inh.
#
# NOTE the grid IS the delay mutation quantum here (a delay mutation is +-1 LEVEL), so this
# tests finer resolution and a halved mutation step together -- they are the same knob.
set -u
SRC=/home/astarostin/projects/spiky/experiments/neurodarwinism/src
BASE=/home/astarostin/projects/spiky/experiments/neurodarwinism/exp012_tiny-direct-genome
PY=/home/astarostin/projects/spiky/.venv/bin/python
export TRITON_CACHE_DIR=/tmp/triton_cache MPLCONFIGDIR=/tmp/mpl
cd "$SRC" || exit 1

FINE=$(seq -s, 1 64)

$PY -u tiny_run_full.py \
  --seeds 1 --pool 512 --cull 64 --rounds 1700 --batch 1024 --ckpt-every 50 \
  --crossover --runner tiny_grow_evolve.py --tag-prefix P \
  --task 555d80d4 \
  --extra "--lam 0 --mu 0 --p-grow 0 --p-shrink 0 --p-affine 0.25 --random-init \
--init-exc 40 --init-inh 10 --p-init 0.10 --quantized --weight-levels=-1.0,0,1.0 \
--delay-levels=$FINE --fanout-cap 16 --inhibition-coeff-evolve --gain-evolve \
--max-episode-batch 128 --readout diagls --out-per-target 8 --out-agg mean" \
  --out-dir "$BASE/run_diagls_k8_finedelay" \
  --label "exp012 K=8 diagonal-LS, FINE delay grid (all 64 ticks) vs odd-only 25.74"
