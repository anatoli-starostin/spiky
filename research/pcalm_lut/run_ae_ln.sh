#!/usr/bin/env bash
# LayerNorm arm for the non-residual autoencoder (task 736f4804 part 2). 2000 steps, L=2, Adam 1e-3,
# no skip. Existing defaults untouched; every run here is a new labelled arm.
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
      --optimizer adam --lr 1e-3 --no-residual --out-dir runs_autoencoder --name "$name" "$@" \
      > "runs_autoencoder_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

# the three stabilisers at tph=128
run "lut-L2-tph128-nores-ln-s2000"  --kind lut --depth-L 2 --tables 128 --block-norm layernorm
# does the "more tables helps without a skip" reversal survive LayerNorm, and extend to 256?
run "lut-L2-tph64-nores-ln-s2000"   --kind lut --depth-L 2 --tables 64  --block-norm layernorm
run "lut-L2-tph256-nores-ln-s2000"  --kind lut --depth-L 2 --tables 256 --block-norm layernorm
# dense control with the same normalisation
run "mlp-L2-nores-ln-s2000"         --kind mlp --depth-L 2 --block-norm layernorm
run "mlpwide-L2-h8192-nores-ln-s2000" --kind mlp --depth-L 2 --hidden 8192 --block-norm layernorm
# and the gain arm at tph=256, so the table axis is complete under both stabilisers
run "lut-L2-tph256-noresgn-s2000"   --kind lut --depth-L 2 --tables 256 --block-norm gain
echo "== AE LN DONE $(date +%T)"
