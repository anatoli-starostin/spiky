#!/usr/bin/env bash
# 2000-step autoencoder runs (task 261cf081). Anatoli explicitly authorised the longer run for this job,
# overriding the standing 500-step cap. Names carry -s2000 so the 500-step artifacts are untouched.
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
      --optimizer adam --lr 1e-3 --out-dir runs_autoencoder --name "$name" "$@" \
      > "runs_autoencoder_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

run "lut-L2-tph128-s2000"    --kind lut    --depth-L 2 --tables 128
run "lut-L2-tph64-s2000"     --kind lut    --depth-L 2 --tables 64
run "linear-adam-s2000"      --kind linear --depth-L 2
run "mlpwide-L2-h8192-s2000" --kind mlp    --depth-L 2 --hidden 8192
run "mlp-L2-adam-s2000"      --kind mlp    --depth-L 2
echo "== AE LONG DONE $(date +%T)"
