#!/usr/bin/env bash
# Launch the 48K run: preflight (refuses on any FAIL), then train.py detached (survives logout / ssh drop),
# stdout+stderr -> train.log in this folder. Prints the PID. Then run verify_wandb.py (see README).
#
#   cp run.env.template run.env && $EDITOR run.env     # fill every FILL
#   bash launch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
[ -f run.env ] || { echo "launch.sh: run.env missing (cp run.env.template run.env and fill it)"; exit 2; }
set -a; . ./run.env; set +a
unset WANDB_MODE                                     # never inherit a disabled/offline mode from the shell
[ -f kernel_gate.json ] || { echo "launch.sh: run the kernel validation gate first: \"\$PYTHON\" validate_kernel.py (see README)"; exit 2; }
[ -z "${WANDB_ENTITY:-}" ] && unset WANDB_ENTITY     # empty -> the key's default entity
[ ! -e train.log ] && [ ! -e metrics.csv ] || { echo "launch.sh: this folder already holds a run (train.log/metrics.csv). Never overwrite a run: use a fresh copy of the package."; exit 2; }

"$PYTHON" preflight.py

export PYTHONUNBUFFERED=1
nohup setsid "$PYTHON" -u train_launch.py > train.log 2>&1 < /dev/null &    # train.py unchanged + the [p2_int8] implementation line
PID=$!
echo "$PID" > train.pid
echo "launched train.py, PID $PID, log $HERE/train.log"
echo "next: wait ~2 min, then:  grep -E '\[wandb\]|\[p2_int8\]' train.log   (wandb must say online; p2_int8 must match preflight_implementation.txt)"
echo "then, after step 500 (~3-4 min): $PYTHON verify_wandb.py"
