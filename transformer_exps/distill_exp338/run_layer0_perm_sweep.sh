#!/bin/bash
set -e
cd /home/starost/spiky/transformer_exps/distill_exp338
for cfg in candidates_l0_perm/p*/config.json; do
  name=$(basename $(dirname $cfg))
  if [ -f "$(dirname $cfg)/summary.json" ]; then echo "[skip] $name"; continue; fi
  echo "=== [$(date +%H:%M:%S)] $name ==="
  /home/starost/spiky/.venv/bin/python -u train_candidate.py "$cfg" \
    --layer 0 --steps 50000 --log-every 10000 \
    --lr 1e-3 --batch-size 1024 --schedule warmup_cosine \
    2>&1 | grep -E "(bit LUTs|best_sign_acc|final_sign_acc)" | head -3
done
echo "=== done ==="
