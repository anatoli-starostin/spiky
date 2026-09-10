#!/usr/bin/env bash
# ONE 4K-native ladder point, H = $1: a 30-step smoke test (fit + speed), then the full run.
# Same protocol as run_ladder_heads.sh. Used to (re)run a point on its own, e.g. H=32 after
# `--student-chunks auto` fixed its 2^31-element embedding_bag fault.
#
#   WAIT_PID=<pid> nohup bash run_ladder_point.sh 32 >> runs/ladder4k_heads.log 2>&1 &
set -u
H=$1
D=/home/astarostin/projects/spiky/experiments/ffn_replacement/distill
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1

while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
if nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; then
    echo "GPU already has a python job -- refusing to start H$H"; exit 1
fi

echo "== smoke H$H $(date)"
if ! $PY -u distill_ffn.py --out "/tmp/distill_smoke_H${H}_chunked" --steps 4000 \
        --max-steps-smoke 30 --eval-every 1000000 --student-overrides "{\"lut_n_heads\": $H}"; then
    echo "== SMOKE FAILED H$H $(date)"; exit 1
fi
OUT=runs/ladder4k_H$H
mkdir -p "$OUT"
echo "== H$H start $(date)"
$PY -u distill_ffn.py --out "$OUT" --steps 4000 \
    --student-overrides "{\"lut_n_heads\": $H}" > "$OUT/train.log" 2>&1
echo "== H$H exit $? $(date)"
