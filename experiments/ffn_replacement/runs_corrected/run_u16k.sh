#!/bin/bash
# Run the two full-length untying runs sequentially on gpustar's 5090, then score each.
# Sequential because each needs ~20 GiB and two would not fit in 31.35 GiB — and because
# nebius's three pending 16k runs must not be disturbed, so nothing goes to the H100.
set -u
RC="$(cd "$(dirname "$0")" && pwd)"
PY=/home/astarostin/projects/spiky/.venv/bin/python
export MPLCONFIGDIR=/tmp/mplcfg PYTORCH_ALLOC_CONF=expandable_segments:True

RUNS=$("$PY" -c "
import json;print(' '.join(r['run'] for r in json.load(open('$RC/u16k_manifest.json'))['runs']))")

for r in $RUNS; do
  D="$RC/$r"
  if [ -f "$D/corrected_score.json" ]; then
    echo "$(date -Is) SKIP $r (already scored)"; continue
  fi
  echo "$(date -Is) === $r : SMOKE param check"
  want=$("$PY" -c "
import json;m=json.load(open('$RC/u16k_manifest.json'))
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
  # --keep: these checkpoints stay on gpustar (gitignored) so the run can be re-scored
  # without refetching; the guard inside still refuses to publish on a dirty load.
  "$PY" "$RC/score_old_grid.py" "$r" --keep > "$D/score.log" 2>&1 \
    || echo "*** scoring failed for $r"
  tail -12 "$D/score.log"
  echo "$(date -Is) === $r : done"
done
echo "$(date -Is) U16K COMPLETE"
