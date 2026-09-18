#!/usr/bin/env bash
# Generic sequential queue for distill_ffn.py runs (one GPU job at a time). Each line of the queue file:
#   <run_name>|<extra distill_ffn.py args, e.g. --steps 4000 --menu-eps 1e-16>|<student-overrides JSON>
# Blank lines / lines starting with # are skipped; a run whose results.json exists is skipped.
#
#   nohup setsid bash run_queue.sh queue_file.txt > runs/<queue>.log 2>&1 &
set -u
D=/home/astarostin/projects/spiky/experiments/ffn_replacement/distill
PY=/home/astarostin/projects/spiky/.venv/bin/python
Q=$(readlink -f "$1")
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill MPLCONFIGDIR=/tmp/mpl
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue ;; esac
  NAME=${line%%|*}; rest=${line#*|}; ARGS=${rest%%|*}; OV=${rest#*|}
  OUT=runs/$NAME
  if [ -e "$OUT/results.json" ]; then echo "== $NAME already has results, skipping"; continue; fi
  while nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 20; done
  mkdir -p "$OUT"
  echo "== $NAME start $(date) args [$ARGS] overrides $OV"
  # shellcheck disable=SC2086
  $PY -u distill_ffn.py --out "$OUT" --save-students $ARGS --student-overrides "$OV" > "$OUT/train.log" 2>&1
  echo "== $NAME exit $? $(date)"
done < "$Q"
echo "== QUEUE DONE $(date)"
