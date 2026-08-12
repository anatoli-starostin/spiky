#!/bin/bash
# exp012: can the tiny spiking net learn ONE of the teacher's sign-comparison bits?
#
#   target  b* = 1[ x_norm[0] > x_norm[16] ]
#   net     17 in -> 8 excitatory -> 1 out, no inhibition, 26 neurons
#
# The teacher's action depends on its input only through 192 bits of exactly this form. Under
# the latency encoder larger x fires EARLIER, so b* is "which of these two input spikes
# arrives first" -- an arrival-ORDER question, which is what delays are for. Delays are
# evolved here and that is the whole point of the probe.
#
# b* is balanced on BOTH splits (P(1) train 0.4796, held-out 0.5080), so its chance of 0.2499
# is meaningful. That mattered: the first bit I picked was 50/50 on training and 12/88 on
# held-out, because the held-out split is the tail of the rollout rather than an i.i.d. draw.
set -u
SRC=/home/astarostin/projects/spiky/experiments/neurodarwinism/src
BASE=/home/astarostin/projects/spiky/experiments/neurodarwinism/exp012_tiny-direct-genome
PY=/home/astarostin/projects/spiky/.venv/bin/python
export TRITON_CACHE_DIR=/tmp/triton_cache MPLCONFIGDIR=/tmp/mpl
cd "$SRC" || exit 1

$PY -u tiny_run_full.py \
  --seeds 1 --pool 512 --cull 64 --rounds 1700 --batch 1024 --ckpt-every 50 \
  --crossover --runner tiny_grow_evolve.py --tag-prefix C \
  --task 85d664f4 \
  --extra "--lam 0 --mu 0 --p-grow 0 --p-shrink 0 --p-affine 0.25 --random-init \
--hidden-capacity 8,0 --init-exc 8 --init-inh 0 --p-init 0.5 --quantized \
--weight-levels=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --p-weight 0.25 \
--delay-levels odd --gain-evolve --max-episode-batch 512 --readout diagls \
--out-per-target 1 --out-agg mean --bit-task 0,16" \
  --out-dir "$BASE/run_bit_0v16" \
  --label "exp012 TINY 17-8-1, sign-comparison bit x0>x16 (chance 0.2499)"
echo "BIT RUN COMPLETE"
