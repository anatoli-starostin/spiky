#!/bin/bash
# Wait for exp139 to finish, then run exp140, then exp141
PY=/home/starost/spiky/.venv/bin/python

# wait for exp139
while pgrep -f "exp139_e64_qk_nap6/train.py" > /dev/null; do sleep 30; done
sleep 10

cd /home/starost/spiky/nanochat_exps/exp140_e64_v_nap6
nohup $PY -u train.py > stdout.log 2>&1
sleep 10

cd /home/starost/spiky/nanochat_exps/exp141_e64_out_nap6
nohup $PY -u train.py > stdout.log 2>&1
