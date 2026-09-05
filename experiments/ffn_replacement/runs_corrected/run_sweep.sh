#!/bin/bash
# Run the 11 proxy-sweep runs sequentially, in the order given by sweep_manifest.json.
# For each: SMOKE=1 param check -> train -> score. Local only, no network, no git.
set -u
RC="$(cd "$(dirname "$0")" && pwd)"
PY=/home/astarostin/projects/spiky/.venv/bin/python
export MPLCONFIGDIR=/tmp/mplcfg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNS=$("$PY" -c "
import json;print(' '.join(r['run'] for r in json.load(open('$RC/sweep_manifest.json'))['runs']))")

for r in $RUNS; do
  D="$RC/$r"
  if [ -f "$D/corrected_score.json" ]; then
    echo "$(date -Is) SKIP $r (already scored)"; continue
  fi
  echo "$(date -Is) === $r : SMOKE param check"
  want=$("$PY" -c "
import json;m=json.load(open('$RC/sweep_manifest.json'))
print(next(x['params'] for x in m['runs'] if x['run']=='$r'))")
  got=$(cd "$D" && SMOKE=1 "$PY" train.py 2>/dev/null | sed -n 's/.*params=\([0-9,]*\).*/\1/p' | tr -d ,)
  if [ "$got" != "$want" ]; then
    echo "$(date -Is) *** ABORT $r: built $got != manifest $want ***"; continue
  fi
  echo "$(date -Is) === $r : params $got OK, training"
  (cd "$D" && "$PY" -u train.py > "$D/train.log" 2>&1)
  if [ ! -f "$D/summary.json" ]; then
    echo "$(date -Is) *** $r FAILED (no summary.json) ***"; tail -5 "$D/train.log"; continue
  fi
  echo "$(date -Is) === $r : scoring"
  "$PY" "$RC/score_sweep.py" "$r" > "$D/score.log" 2>&1 || echo "*** scoring failed for $r"
  echo "$(date -Is) === $r : done"
done
echo "$(date -Is) SWEEP COMPLETE"
