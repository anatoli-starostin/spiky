#!/usr/bin/env bash
# 4K-NATIVE capacity ladder over lut_n_heads (H = 4, 8, 16 -> 1x, 2x, 4x params).
# Everything except H is the exp_n_0238 student and the sweep_0238arch_16k protocol: same data
# order, batch, eval cadence, held-out slab and linear baseline; --steps 4000 makes the cosine
# schedule span the 4K steps. Runs strictly one job at a time.
#
# H=32 (8x, 51.5M) was dropped at Anatoly's request (task 38fd48ea). It is not a size limit:
# with `--student-chunks auto` it would step in 3 chunks; run_ladder_point.sh 32 does one point.
#
#   WAIT_PID=<pid of a running job> nohup bash run_ladder_heads.sh > runs/ladder4k_heads.log 2>&1 &
set -u
D=/home/astarostin/projects/spiky/experiments/ffn_replacement/distill
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1

while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
if nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; then
    echo "GPU already has a python job -- refusing to start"; exit 1
fi

for H in 4 8 16; do
    OUT=runs/ladder4k_H$H
    mkdir -p "$OUT"
    echo "== H$H start $(date)"
    $PY -u distill_ffn.py --out "$OUT" --steps 4000 \
        --student-overrides "{\"lut_n_heads\": $H}" > "$OUT/train.log" 2>&1
    echo "== H$H exit $? $(date)"
done
echo "== LADDER DONE $(date)"
