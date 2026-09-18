#!/usr/bin/env bash
# Matrix-menu tph sweep (+ Light controls), 4K-native, same protocol as runs/ladder4k_H8 (not rerun).
# Sequential, one run at a time on the GPU. Each run: runs/<name>/{train.log, curves.csv, results.json, manifest.json,
# student_L*.pt (gitignored)}.
#
#   nohup setsid bash run_menu_tph_sweep.sh > runs/menu_tph_sweep.log 2>&1 &
set -u
D=/home/astarostin/projects/spiky/experiments/ffn_replacement/distill
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill

MENU='"lut_cell_mode": "matrix_menu", "lut_menu_size": 64, "lut_menu_forward": "hard"'
RUNS=(
  "menu4k_H8_tph128|{\"lut_n_heads\": 8, \"lut_tables_per_head\": 128, $MENU}"
  "menu4k_H8_tph32|{\"lut_n_heads\": 8, \"lut_tables_per_head\": 32, $MENU}"
  "menu4k_H8_tph16|{\"lut_n_heads\": 8, \"lut_tables_per_head\": 16, $MENU}"
  "menu4k_H8_tph8|{\"lut_n_heads\": 8, \"lut_tables_per_head\": 8, $MENU}"
  "light4k_H8_tph32|{\"lut_n_heads\": 8, \"lut_tables_per_head\": 32}"
  "light4k_H8_tph8|{\"lut_n_heads\": 8, \"lut_tables_per_head\": 8}"
)
for r in "${RUNS[@]}"; do
  NAME=${r%%|*}; OV=${r#*|}
  OUT=runs/$NAME
  if [ -e "$OUT/results.json" ]; then echo "== $NAME already has results, skipping"; continue; fi
  if nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; then
    echo "== GPU already has a python job -- refusing to start $NAME $(date)"; exit 1
  fi
  mkdir -p "$OUT"
  echo "== $NAME start $(date) overrides $OV"
  $PY -u distill_ffn.py --out "$OUT" --steps 4000 --save-students --student-overrides "$OV" > "$OUT/train.log" 2>&1
  echo "== $NAME exit $? $(date)"
done
echo "== SWEEP DONE $(date)"
