#!/usr/bin/env bash
# Refresh the Slack progress bar once a minute while the LUT sep-CMA-ES run is alive.
#
# Exits as soon as the training PID is gone, deliberately: the harness re-invokes the
# agent when this exits, which is how "tell me when the run has ended" is delivered
# without polling. It does NOT touch the bar after that -- the last state stays up.
#
# progress.py only writes files under ~/.cache/slack_facade/progress (the green-zone
# rendezvous the face reads), so this whole loop is cage-safe and needs no network.
set -u

PID="${1:?usage: bar_updater.sh <training-pid>}"
HANDLE=0ab5d40d
LOG=/home/astarostin/projects/spiky/experiments/walker2d_lut/exp_c24_sepcma/sepcma_lut_run.log
PROG=/home/astarostin/work/slack-facade/progress.py
SPG=28.0            # measured s/gen, steady across both arms
TOTAL=600

while kill -0 "$PID" 2>/dev/null; do
    # The trainer logs every 10th generation, so G moves in steps of 10 (~4.7 min).
    # Updating every 60 s is still worth it: it keeps the bar's "minutes left" honest
    # even between fitness refreshes.
    line=$(grep " gen " "$LOG" | tail -1)
    if [ -n "$line" ]; then
        G=$(sed -E 's#.*gen +([0-9]+)/.*#\1#' <<<"$line")
        MEAN=$(sed -E 's#.*mean-policy +([0-9.-]+).*#\1#' <<<"$line")
        POPBEST=$(sed -E 's#.*pop best +([0-9.-]+).*#\1#' <<<"$line")
        BESTMEAN=$(sed -E 's#.*\(best mean +([0-9.-]+)\).*#\1#' <<<"$line")
        ETA=$(awk -v g="$G" -v t="$TOTAL" -v s="$SPG" 'BEGIN{printf "%d", (t-g)*s/60}')
        python3 "$PROG" update "$HANDLE" --pct 75 --stats \
"step 4 running: sep-CMA-ES on the LUT (nap 6, tph 16, d=7,872), gen ${G}/600, ~${ETA} min left. Live MJX fitness ~${MEAN} (best mean ${BESTMEAN}, pop best ${POPBEST}). Currently the best gradient-free result in the chapter — ~2.6x the old exp_c05 LUT (879.9), ~1.4x the exp_c24 MLP (1617.5). Final 100-episode CPU eval pending." \
            >/dev/null 2>&1
    fi
    sleep 60
done

echo "LUT RUN ENDED: pid $PID is gone at $(date -u +%FT%TZ)"
grep " gen " "$LOG" | tail -2
tail -2 "$LOG"
