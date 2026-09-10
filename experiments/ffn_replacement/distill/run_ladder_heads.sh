#!/usr/bin/env bash
# 4K-NATIVE capacity ladder over lut_n_heads (H = 4, 8, 16, 32 -> 1x, 2x, 4x, 8x params).
# Everything except H is the exp_n_0238 student and the sweep_0238arch_16k protocol: same data
# order, batch, eval cadence, held-out slab and linear baseline; --steps 4000 makes the cosine
# schedule span the 4K steps. Runs strictly one job at a time.
#
#   WAIT_PID=<pid of a running job> nohup bash run_ladder_heads.sh > runs/ladder4k_heads.log 2>&1 &
set -u
D=/home/astarostin/projects/spiky/experiments/ffn_replacement/distill
PY=/home/astarostin/projects/spiky/.venv/bin/python
SMOKE=${SMOKE_DIR:-/tmp/distill_smoke_H32}
cd "$D" || exit 1

while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
if nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; then
    echo "GPU already has a python job -- refusing to start"; exit 1
fi

# The largest point first, for 30 steps: does it fit with all 6 students, and how fast is it?
echo "== smoke H32 $(date)"
if $PY -u distill_ffn.py --out "$SMOKE" --steps 4000 --max-steps-smoke 30 --eval-every 1000000 \
       --student-overrides '{"lut_n_heads": 32}'; then H32_OK=1; else H32_OK=0; fi

for H in 4 8 16 32; do
    if [ "$H" = 32 ] && [ "$H32_OK" = 0 ]; then
        echo "== SKIP H32: smoke failed (see above), needs layer grouping"; continue
    fi
    OUT=runs/ladder4k_H$H
    mkdir -p "$OUT"
    echo "== H$H start $(date)"
    $PY -u distill_ffn.py --out "$OUT" --steps 4000 \
        --student-overrides "{\"lut_n_heads\": $H}" > "$OUT/train.log" 2>&1
    echo "== H$H exit $? $(date)"
done
echo "== LADDER DONE $(date)"
