#!/bin/bash
# Run BitPermutationLUT exp301-303 quality sweep sequentially.
# (exp300 bit_lut_lr=3e-3 was killed -- clearly worse than exp299 baseline.)
cd "$(dirname "$0")/.."

for exp in exp301_bit_lr_1e2 exp302_init_std_0_01 exp303_beta2_0_99; do
  echo "=== $(date -Is) launching $exp ==="
  .venv/bin/python -u "transformer_exps/$exp/train.py" \
    > "transformer_exps/$exp/stdout.log" 2>&1 || echo "    $exp exited $?"
  echo "=== $(date -Is) finished $exp ==="
done
echo "=== sweep done ==="
