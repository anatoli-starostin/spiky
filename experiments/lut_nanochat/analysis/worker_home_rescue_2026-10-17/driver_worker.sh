#!/bin/bash
# Self-contained worker driver: train R6 then S2 (16k each, bs24/ga2), score under the
# corrected protocol, commit+push results to research/ffn_replacement_fix. Detached.
set -u
export TZ=UTC
export NANOCHAT_ROOT=$HOME/projects/nanochat
export MPLCONFIGDIR=/tmp/mpl TRITON_CACHE_DIR=/tmp/triton PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SP=$HOME/projects/spiky
RC=$SP/experiments/ffn_replacement/runs_corrected
TOOLS=$SP/experiments/ffn_replacement/tools
PY=$HOME/venv/bin/python
RUNS=("exp_n_0166_r6real_compmhl_H2_din64_dout48_tph512" "exp_n_0167_s2real_compmhl_H4_din64_dout32_tph256")
for R in "${RUNS[@]}"; do
  D=$RC/$R
  cd "$D"
  : > run.log
  echo "=== START $R $(date -u +%H:%M:%S) ===" >> run.log
  $PY -u train.py >> run.log 2>&1
  echo "=== TRAIN EXIT $? $(date -u +%H:%M:%S) ===" >> run.log
  if grep -q "=== DONE ===" run.log; then
    $PY "$TOOLS/score_checkpoint.py" --run "$D" > corrected_score.json 2>> run.log || echo "SCORE_FAILED" >> run.log
    cd "$SP"
    git pull --rebase origin research/ffn_replacement_fix >> "$D/run.log" 2>&1 || true
    git add "experiments/ffn_replacement/runs_corrected/$R/config.json" \
            "experiments/ffn_replacement/runs_corrected/$R/train.py" \
            "experiments/ffn_replacement/runs_corrected/$R/metrics.csv" \
            "experiments/ffn_replacement/runs_corrected/$R/summary.json" \
            "experiments/ffn_replacement/runs_corrected/$R/loss.png" \
            "experiments/ffn_replacement/runs_corrected/$R/corrected_score.json" >> "$D/run.log" 2>&1
    git commit -m "runs_corrected: $R results under corrected protocol (worker) (#111)" \
               -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" >> "$D/run.log" 2>&1 || true
    git push origin research/ffn_replacement_fix >> "$D/run.log" 2>&1 && echo "=== PUSHED $R $(date -u +%H:%M:%S) ===" >> "$D/run.log" || echo "=== PUSH_FAILED $R ===" >> "$D/run.log"
  else
    echo "=== $R DID NOT FINISH CLEANLY — skip score/commit ===" >> "$D/run.log"
  fi
done
echo "ALL_WORKER_RUNS_DONE $(date -u +%H:%M:%S)" >> $RC/driver_worker.done
