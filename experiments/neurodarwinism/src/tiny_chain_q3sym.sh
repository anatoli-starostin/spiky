#!/bin/bash
# Wait for the quantized fanout-16 run to finish, capture its final numbers, then launch the
# symmetric-grid + inhibition-coefficient run. Written as a script so the whole sequence is
# one detached process and the launch cannot be missed.
set -u
SRC=/home/astarostin/projects/spiky/experiments/neurodarwinism/src
BASE=/home/astarostin/projects/spiky/experiments/neurodarwinism/exp012_tiny-direct-genome
PY=/home/astarostin/projects/spiky/.venv/bin/python
export TRITON_CACHE_DIR=/tmp/triton_cache MPLCONFIGDIR=/tmp/mpl

cd "$SRC" || exit 1

echo "waiting for the quantized fanout-16 run to finish..."
until [ -f "$BASE/full_run_quantized_fanout16_random/z0_final.json" ]; do sleep 30; done
echo "it finished; evaluating its leader"
$PY -u tiny_grow_final_eval.py \
    --ckpt "$BASE/full_run_quantized_fanout16_random/ck_z0.npz" --lam 0 --mu 0 \
    --out "$BASE/full_run_quantized_fanout16_random/final_leader.json" \
    > /tmp/exp012_z_eval.log 2>&1
echo "eval done; launching the symmetric-grid run"

# solo now, so the 256 chunk is safe again
$PY -u tiny_run_full.py --runner tiny_grow_evolve.py --seeds 1 \
    --pool 512 --cull 64 --rounds 1700 --batch 1024 --ckpt-every 50 --crossover \
    --extra "--lam 0 --mu 0 --p-grow 0 --p-shrink 0 --p-affine 0.25 --random-init \
--init-exc 40 --init-inh 10 --p-init 0.10 --quantized --weight-levels=-1.0,0,1.0 \
--delay-levels odd --fanout-cap 16 --inhibition-coeff-evolve --gain-evolve \
--max-episode-batch 256" \
    --tag-prefix y --out-dir "$BASE/full_run_q3sym_coeff_random" \
    --task 75220d01 --label "exp012 symmetric 3-value grid + inhibition coeff, RANDOM" \
    > /tmp/exp012_ydriver.log 2>&1
