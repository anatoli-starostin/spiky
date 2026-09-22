#!/usr/bin/env bash
# LUT autoencoder sweep (task ec7205ef). 500 steps, seed 0, Fashion-MNIST, local logging only.
# Adam 1e-3 is the headline -- Adam is what makes BP work on this architecture -- with one SGD run for
# contrast. The linear 784->64->784 run is the PCA-equivalent bottleneck baseline and the MLP runs are
# the matched control: identical shape, residual MLP blocks instead of LightMHL.
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
  $PY -u autoencoder.py --steps 500 --batch 128 --width 64 --tables 64 --seed 0 --probe-every 25 \
      --out-dir runs_autoencoder --name "$name" "$@" > "runs_autoencoder_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

for L in 2 4 8; do
  run "lut-L$L-adam"  --kind lut --depth-L $L --optimizer adam --lr 1e-3
  run "mlp-L$L-adam"  --kind mlp --depth-L $L --optimizer adam --lr 1e-3
done
run "lut-L4-sgd3e-2"  --kind lut --depth-L 4 --optimizer sgd  --lr 3e-2
run "mlp-L4-sgd3e-2"  --kind mlp --depth-L 4 --optimizer sgd  --lr 3e-2
run "linear-adam"     --kind linear --depth-L 2 --optimizer adam --lr 1e-3
echo "== AE GRID DONE $(date +%T)"
