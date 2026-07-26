#!/bin/bash
set -u
PY=/home/astarostin/projects/spiky/.venv/bin/python
B=/home/astarostin/projects/spiky/experiments/hyperplane_ffn
declare -A CK=( [exp023]=$B/exp023_singlestream_linear_unembedder/checkpoint.pt
                [exp024]=$B/exp024_singlestream_linear_anchorinit/checkpoint.pt )
declare -A CFG=( [exp023]=$B/exp023_singlestream_linear_unembedder/config.json
                 [exp024]=$B/exp024_singlestream_linear_anchorinit/config.json )
one(){
  run=$1; mode=$2; lut=$3
  D=/tmp/fe2_${run}_${mode}; rm -rf "$D"; mkdir -p "$D"; cp /tmp/exp023_fe2/train.py "$D/train.py"
  "$PY" - "${CFG[$run]}" "$D" "$lut" <<'PY'
import json,sys
cfg,D,lut=sys.argv[1],sys.argv[2],sys.argv[3]
c=json.load(open(cfg)); c["lut_layer_type"]=lut; c["n_steps"]=1; c["eval_steps"]=20; c["exp_name"]="fe2"
json.dump(c,open(D+"/config.json","w"),indent=2)
PY
  cd "$D"
  echo "----- $run / $mode (lut=$lut) -----"
  SWAP_EVAL=1 SWAP_MODE=$mode SWAP_CKPT="${CK[$run]}" NANOCHAT_ROOT=/home/astarostin/projects/nanochat PYTHONPATH=/home/astarostin/projects/nanochat MPLCONFIGDIR=/tmp/mpl "$PY" -u train.py 2>&1 | grep -E "LOAD |FLIP_|RESULT |Error|Traceback|out of memory" | head -12
}
for r in exp023 exp024; do
  one $r baseline hyperplane
  one $r argmaxmin fast
done
