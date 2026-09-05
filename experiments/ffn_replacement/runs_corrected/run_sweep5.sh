#!/bin/bash
# Run the 6 second-sweep runs sequentially, in sweep5_manifest.json order.
# Per run: SMOKE=1 param check -> train (falling back to device_batch 6 x accum 8 if the
# bs12 attempt dies without a summary) -> score. Local only, no network, no git.
set -u
RC="$(cd "$(dirname "$0")" && pwd)"
PY=/home/astarostin/projects/spiky/.venv/bin/python
export MPLCONFIGDIR=/tmp/mplcfg
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUNS=$("$PY" -c "
import json;print(' '.join(r['run'] for r in json.load(open('$RC/sweep5_manifest.json'))['runs']))")

for r in $RUNS; do
  D="$RC/$r"
  if [ -f "$D/corrected_score.json" ]; then
    echo "$(date -Is) SKIP $r (already scored)"; continue
  fi
  echo "$(date -Is) === $r : SMOKE param check"
  want=$("$PY" -c "
import json;m=json.load(open('$RC/sweep5_manifest.json'))
print(next(x['params'] for x in m['runs'] if x['run']=='$r'))")
  got=$(cd "$D" && SMOKE=1 "$PY" train.py 2>/dev/null | sed -n 's/.*params=\([0-9,]*\).*/\1/p' | tr -d ,)
  if [ "$got" != "$want" ]; then
    echo "$(date -Is) *** ABORT $r: built $got != manifest $want ***"; continue
  fi
  echo "$(date -Is) === $r : params $got OK, training at device_batch 12"
  (cd "$D" && "$PY" -u train.py > "$D/train.log" 2>&1)

  if [ ! -f "$D/summary.json" ]; then
    # bs12 did not finish. If it was memory, retry once at 6 x 4 (same effective batch 24
    # sequences, so the run stays comparable); anything else will fail the same way again.
    echo "$(date -Is) === $r : bs12 attempt produced no summary, retrying at device_batch 6"
    tail -3 "$D/train.log"
    mv "$D/train.log" "$D/train_failed_bs12.log"
    "$PY" - "$D/config.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c['device_batch_size'] = 6
c['_mem_note'] = ('device_batch 6 / grad_accum 8 after the bs12 attempt failed on this 5090 '
                  '(see train_failed_bs12.log). Effective batch unchanged at 24 sequences '
                  '/ 12,288 tokens, and eval is decoupled from device_batch_size, so the run '
                  'stays fully comparable with the rest of the sweep.')
json.dump(c, open(p, 'w'), indent=2)
PYEOF
    (cd "$D" && "$PY" -u train.py > "$D/train.log" 2>&1)
  fi

  if [ ! -f "$D/summary.json" ]; then
    echo "$(date -Is) *** $r FAILED at both batch sizes ***"; tail -5 "$D/train.log"; continue
  fi
  echo "$(date -Is) === $r : scoring"
  "$PY" "$RC/score_sweep.py" "$r" > "$D/score.log" 2>&1 || echo "*** scoring failed for $r"
  echo "$(date -Is) === $r : done"
done
echo "$(date -Is) SWEEP5 COMPLETE"
