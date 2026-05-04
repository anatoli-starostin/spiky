#!/bin/bash
PY=/home/starost/spiky/.venv/bin/python
while kill -0 3452718 2>/dev/null; do sleep 30; done
sleep 10
cd /home/starost/spiky/nanochat_exps/exp146_d2v_before_sdpa
nohup $PY -u train.py > stdout.log 2>&1
