#!/bin/bash
set -u
PY=/home/astarostin/projects/spiky/.venv/bin/python
SRC=/home/astarostin/projects/spiky/experiments/hyperplane_ffn/exp024_singlestream_linear_anchorinit
CK=$SRC/checkpoint.pt
one(){
  k=$1
  D=/tmp/trunc_$k; rm -rf "$D"; mkdir -p "$D"; cp /tmp/exp024_trunc/train.py "$D/train.py"
  "$PY" - "$SRC/config.json" "$D" <<'PY'
import json,sys
cfg,D=sys.argv[1],sys.argv[2]
c=json.load(open(cfg)); c["n_steps"]=1; c["eval_steps"]=20; c["exp_name"]="trunc"
json.dump(c,open(D+"/config.json","w"),indent=2)
PY
  cd "$D"
  echo "----- k=$k -----"
  TRUNC_EVAL=1 TRUNC_K=$k TRUNC_CKPT="$CK" NANOCHAT_ROOT=/home/astarostin/projects/nanochat \
    PYTHONPATH=/home/astarostin/projects/nanochat MPLCONFIGDIR=/tmp/mpl \
    "$PY" -u train.py 2>&1 | grep -E "LOAD |NUM_SITES|FLIP_|RESULT |Error|Traceback|out of memory" | head -12
}
for k in full 2 3 4 5; do one $k; done
