#!/bin/bash
set -e
cd /home/starost/spiky/transformer_exps/distill_exp338
for cfg in candidates_l0/l0_*/config.json; do
  name=$(basename $(dirname $cfg))
  sum=$(dirname $cfg)/summary.json
  if [ -f "$sum" ]; then
    echo "[skip] $name"
    continue
  fi
  echo "=== [$(date +%H:%M:%S)] $name ==="
  /home/starost/spiky/.venv/bin/python -u train_candidate.py "$cfg" \
    --layer 0 --steps 50000 --log-every 25000 \
    --lr 1e-3 --batch-size 1024 --schedule warmup_cosine \
    2>&1 | grep -E "(^bit LUTs|best_sign_acc|final_sign_acc)" | head -3
done

# Also run teacher-shape with teacher pairs as a separate ceiling sample.
echo "=== [$(date +%H:%M:%S)] teacher_shape + teacher_pairs ==="
mkdir -p candidates_l0/l0_s_teacher_shape_teacher_pairs
cp candidates_l0/l0_s_teacher_shape/config.json candidates_l0/l0_s_teacher_shape_teacher_pairs/config.json
/home/starost/spiky/.venv/bin/python -u train_candidate.py candidates_l0/l0_s_teacher_shape_teacher_pairs/config.json \
  --layer 0 --steps 50000 --log-every 25000 \
  --lr 1e-3 --batch-size 1024 --schedule warmup_cosine --load-teacher-pairs \
  2>&1 | grep -E "(^bit LUTs|best_sign_acc|final_sign_acc)" | head -3
echo "=== [$(date +%H:%M:%S)] sweep done ==="
