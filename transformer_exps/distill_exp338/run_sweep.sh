#!/bin/bash
set -e
cd /home/starost/spiky/transformer_exps/distill_exp338
for cfg in candidates/c_*/config.json; do
  name=$(basename $(dirname $cfg))
  sum=$(dirname $cfg)/summary.json
  if [ -f "$sum" ]; then
    echo "[skip] $name (already done)"
    continue
  fi
  echo "=== [$(date +%H:%M:%S)] training $name ==="
  /home/starost/spiky/.venv/bin/python -u train_candidate.py "$cfg" \
    --layer 3 --steps 50000 --log-every 25000 \
    --lr 1e-3 --batch-size 1024 --schedule warmup_cosine \
    2>&1 | grep -E "(^bit LUTs|best_sign_acc|final_sign_acc)" | head -3
done
echo "=== [$(date +%H:%M:%S)] sweep done ==="
