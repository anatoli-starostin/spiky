#!/usr/bin/env bash
# Table-count axis for the autoencoder (task 08d26e08), in the priority order given:
#   1. L in {2,4,8} at tph=64 -- already done by run_ae_grid.sh
#   2. tph in {128,256} at the middle depth L=4
#   3. the rest of the cross product, only because step 2 turned out to cost seconds per run
# plus a PARAMETER-MATCHED MLP control, so "more parameters helps" can be told apart from
# "LUT structure helps". A LUT block at tph=64 is 64*256*64 = 1,048,576 table entries; the matched MLP
# block is 64 -> 8192 -> 64 = 1,048,576 weights. Same budget, different structure.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_autoencoder_logs

run () {
  local name=$1; shift
  if [ -e "runs_autoencoder/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u autoencoder.py --steps 500 --batch 128 --width 64 --seed 0 --probe-every 25 \
      --optimizer adam --lr 1e-3 --out-dir runs_autoencoder --name "$name" "$@" \
      > "runs_autoencoder_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

# priority 2: the table axis at the middle depth
for tph in 128 256; do
  run "lut-L4-tph$tph" --kind lut --depth-L 4 --tables $tph
done
# priority 3: the rest of the cross product
for L in 2 8; do
  for tph in 128 256; do
    run "lut-L$L-tph$tph" --kind lut --depth-L $L --tables $tph
  done
done
# parameter-matched MLP controls at the L=4 budget
run "mlpwide-L4-h8192" --kind mlp --depth-L 4 --hidden 8192
run "mlpwide-L2-h8192" --kind mlp --depth-L 2 --hidden 8192
echo "== AE TABLE SWEEP DONE $(date +%T)"
