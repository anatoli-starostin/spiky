#!/bin/bash
# Fetch one checkpoint at a time from nebius-h100, score it under the corrected protocol,
# then delete it. Only the score file is kept — disk is limited.
#
# Needs network, so this runs OUTSIDE the sbox cage (the cage has no network and masks
# ~/.ssh). Scoring itself is the ordinary GPU path.
#
#     bash fetch_score_delete.sh <run> [<run> ...]
#
# A run whose checkpoint is absent on nebius is recorded in unscoreable.json with the reason
# and skipped, never silently dropped.
set -u
RC="$(cd "$(dirname "$0")" && pwd)"
PY=/home/astarostin/projects/spiky/.venv/bin/python
REMOTE='nebius-h100:~/projects/spiky/experiments/hyperplane_ffn'
BAR=$(cat /tmp/oldgrid_bar.txt 2>/dev/null || echo "")
export MPLCONFIGDIR=/tmp/mplcfg PYTORCH_ALLOC_CONF=expandable_segments:True

for r in "$@"; do
  D="$RC/$r"
  if [ -f "$D/corrected_score.json" ]; then
    echo "$(date -Is) SKIP $r (already scored)"; continue
  fi
  echo "$(date -Is) === $r : fetching"
  if ! rsync -a "$REMOTE/$r/checkpoint.pt" "$D/checkpoint.pt" 2>/dev/null; then
    echo "$(date -Is) *** $r : no checkpoint on nebius — recording as unscoreable"
    "$PY" - "$RC" "$r" <<'PYEOF'
import json, os, sys
rc, run = sys.argv[1], sys.argv[2]
p = os.path.join(rc, 'unscoreable.json')
d = json.load(open(p)) if os.path.exists(p) else {}
d[run] = 'checkpoint.pt not present on nebius-h100:~/projects/spiky/experiments/hyperplane_ffn'
json.dump(d, open(p, 'w'), indent=2)
PYEOF
    continue
  fi
  echo "$(date -Is) === $r : scoring ($(stat -c%s "$D/checkpoint.pt") bytes)"
  "$PY" "$RC/score_old_grid.py" "$r" 2>&1 \
    | grep -vE "UserWarning|warnings.warn|AllocatorConfig"
  rc_code=${PIPESTATUS[0]}
  if [ -f "$D/checkpoint.pt" ]; then rm -f "$D/checkpoint.pt"; echo "   checkpoint removed"; fi
  if [ "$rc_code" != "0" ]; then
    echo "$(date -Is) *** $r : scorer exited $rc_code — no number published"
  fi
  if [ -n "$BAR" ]; then
    n=$(ls "$RC"/exp_n_*/corrected_score.json 2>/dev/null | wc -l)
    "$PY" -c "
import sys; sys.path.insert(0, '/home/astarostin/work/slack-facade')
import progress; progress.progress_update('$BAR', step=$n - 7, total=18,
    stats='scored $r; $((n-7))/18 of the in-scope old-grid runs done')" 2>/dev/null || true
  fi
done
echo "$(date -Is) BATCH DONE"
