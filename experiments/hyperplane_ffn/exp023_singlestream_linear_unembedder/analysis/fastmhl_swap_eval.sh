#!/bin/bash
set -u
CK=/home/astarostin/projects/spiky/experiments/hyperplane_ffn/exp023_singlestream_linear_unembedder/checkpoint.pt
CFG=/home/astarostin/projects/spiky/experiments/hyperplane_ffn/exp023_singlestream_linear_unembedder/config.json
PY=/home/astarostin/projects/spiky/.venv/bin/python
run(){
  mode=$1; D=/tmp/exp023_fasteval_$mode; rm -rf "$D"; mkdir -p "$D"; cp /tmp/exp023_fasteval/train.py "$D/train.py"
  "$PY" - "$CFG" "$D" "$mode" <<'PY'
import json,sys
cfg,D,mode=sys.argv[1],sys.argv[2],sys.argv[3]
c=json.load(open(cfg))
c["lut_layer_type"]=mode; c["n_steps"]=1; c["eval_steps"]=20; c["exp_name"]="fasteval_"+mode
json.dump(c,open(D+"/config.json","w"),indent=2)
PY
  cd "$D"
  echo "----- mode=$mode -----"
  SWAP_EVAL=1 SWAP_CKPT="$CK" NANOCHAT_ROOT=/home/astarostin/projects/nanochat PYTHONPATH=/home/astarostin/projects/nanochat MPLCONFIGDIR=/tmp/mpl "$PY" -u train.py 2>&1 | grep -E "LOAD |sample_|SWAP_.*_BPB|Error|Traceback|out of memory" | head -12
}
run hyperplane
run fast
