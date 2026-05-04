#!/bin/bash
PY=/home/starost/spiky/.venv/bin/python

# wait for exp139 (PID 3084787 confirmed running)
while kill -0 3084787 2>/dev/null; do sleep 30; done
sleep 10

cd /home/starost/spiky/nanochat_exps/exp140_e64_v_nap6
nohup $PY -u train.py > stdout.log 2>&1
EXP140_RC=$?
sleep 10

cd /home/starost/spiky/nanochat_exps/exp141_e64_out_nap6
nohup $PY -u train.py > stdout.log 2>&1
