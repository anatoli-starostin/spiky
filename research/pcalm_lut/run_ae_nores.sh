#!/usr/bin/env bash
# Non-residual autoencoder (task b320b253): h = block(h), no identity path, a_i dropped to 1.0 and the
# init calibrated so each block preserves its input RMS. Two families:
#   -nores-s2000     exactly as specified: plain sequential, nothing anchoring the scale
#   -noresgn-s2000   the same plus parameter-free gain normalisation, because the plain form diverges
# The residual -s2000 results are untouched.
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
  $PY -u autoencoder.py --steps 2000 --batch 128 --width 64 --seed 0 --probe-every 100 \
      --optimizer adam --no-residual --out-dir runs_autoencoder --name "$name" "$@" \
      > "runs_autoencoder_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

# as specified, at the sweep lr
run "lut-L2-tph128-nores-s2000"    --kind lut --depth-L 2 --tables 128 --lr 1e-3
run "lut-L2-tph64-nores-s2000"     --kind lut --depth-L 2 --tables 64  --lr 1e-3
run "mlpwide-L2-h8192-nores-s2000" --kind mlp --depth-L 2 --hidden 8192 --lr 1e-3
run "mlp-L2-adam-nores-s2000"      --kind mlp --depth-L 2 --lr 1e-3
# is the divergence just the learning rate? two decades lower, same arm
run "lut-L2-tph128-nores-lr1e-4"   --kind lut --depth-L 2 --tables 128 --lr 1e-4
# with the minimal stabiliser
run "lut-L2-tph128-noresgn-s2000"    --kind lut --depth-L 2 --tables 128 --gain-norm --lr 1e-3
run "lut-L2-tph64-noresgn-s2000"     --kind lut --depth-L 2 --tables 64  --gain-norm --lr 1e-3
run "mlpwide-L2-h8192-noresgn-s2000" --kind mlp --depth-L 2 --hidden 8192 --gain-norm --lr 1e-3
run "mlp-L2-adam-noresgn-s2000"      --kind mlp --depth-L 2 --gain-norm --lr 1e-3
echo "== AE NORES DONE $(date +%T)"
