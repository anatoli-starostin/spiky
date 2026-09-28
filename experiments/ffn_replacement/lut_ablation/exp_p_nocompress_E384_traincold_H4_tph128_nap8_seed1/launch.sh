#!/usr/bin/env bash
# Minimal launcher for the dense vanilla+dropout run (no p2 kernel -> no kernel gate). Sources
# run.env, launches train.py detached (survives logout), stdout+stderr -> train.log. Prints PID.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
[ -f run.env ] || { echo "run.env missing (cp run.env.template? fill it)"; exit 2; }
set -a; . ./run.env; set +a
unset WANDB_MODE
[ -z "${WANDB_ENTITY:-}" ] && unset WANDB_ENTITY
[ ! -e train.log ] && [ ! -e metrics.csv ] || { echo "this folder already holds a run (train.log/metrics.csv); use a fresh copy"; exit 2; }
export PYTHONUNBUFFERED=1
nohup setsid "$PYTHON" -u train.py > train.log 2>&1 < /dev/null &
echo "$!" > train.pid
echo "launched train.py, PID $(cat train.pid), log $HERE/train.log"
echo "then, after step 500: \"$PYTHON\" verify_wandb.py"
