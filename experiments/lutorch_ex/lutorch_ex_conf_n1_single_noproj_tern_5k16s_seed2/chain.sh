#!/bin/bash
# The three arms sequentially in ONE run (continues with the next arm if one fails; exit code = number of failed arms).
D=$(cd "$(dirname "$0")" && pwd); fail=0
for arm in tern4 tern16 ternfree; do
  echo "=== ARM $arm $(date -u +%FT%TZ) ==="
  bash $D/run.sh $arm || { echo "ARM $arm FAILED"; fail=$((fail+1)); }
done
echo "=== chain done, failed arms: $fail ==="; exit $fail
