"""Confirm the live W&B run exists AND metrics are arriving -- run after the first eval (step 500), before walking away.

    set -a; . ./run.env; set +a; "$PYTHON" verify_wandb.py

PASS needs all of:
  * train.log has "[wandb] online: run <exp_name>" (not "off", not "offline", not "unreachable")
  * the run exists on wandb.ai under project Spiky with id = exp_name, state "running"
  * its history has train rows beyond step 100 and at least one val_bpb row, and the latest val_bpb equals the latest
    metrics.csv row (the local record) to 4 decimals
Exit 0 on PASS, 1 otherwise (then stop the run -- kill $(cat train.pid) -- fix W&B, and relaunch from a FRESH package copy;
never delete the W&B run: the server reserves deleted ids).
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
name = json.load(open(os.path.join(HERE, "config.json")))["exp_name"]
ok = True


def say(tag, good, msg):
    global ok
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {tag} -- {msg}", flush=True)


log = open(os.path.join(HERE, "train.log")).read()
lines = [ln for ln in log.splitlines() if ln.startswith("[wandb]")]
say("trainer W&B mode", any(re.match(rf"\[wandb\] online: run {re.escape(name)}\b", ln) for ln in lines), " | ".join(lines[:3]))

rows = list(csv.DictReader(open(os.path.join(HERE, "metrics.csv")))) if os.path.exists(os.path.join(HERE, "metrics.csv")) else []
say("local metrics.csv has an eval row", bool(rows), f"{len(rows)} rows")

try:
    import wandb
    api = wandb.Api(timeout=30)
    entity = os.environ.get("WANDB_ENTITY") or api.viewer.entity
    run = api.run(f"{entity}/Spiky/{name}")
    say("run exists on wandb.ai", True, f"project Spiky, id {run.id}, state {run.state}, url {run.url}")
    say("run state running", run.state == "running", run.state)
    hist = list(run.scan_history(keys=["_step", "train/loss"]))
    say("train metrics arriving", len(hist) > 0 and max(h["_step"] for h in hist) >= 100,
        f"{len(hist)} train rows, last step {max((h['_step'] for h in hist), default=None)}")
    vh = list(run.scan_history(keys=["_step", "val_bpb"]))
    say("val_bpb arriving", len(vh) > 0, f"{len(vh)} val rows")
    if vh and rows:
        last_w, last_l = vh[-1], rows[-1]
        say("W&B val_bpb == metrics.csv", abs(float(last_w["val_bpb"]) - float(last_l["val_bpb"])) < 5e-5 and int(last_w["_step"]) == int(last_l["step"]),
            f"wandb step {last_w['_step']} val_bpb {last_w['val_bpb']:.6f} vs csv step {last_l['step']} val_bpb {last_l['val_bpb']}")
except Exception as e:
    say("wandb.ai query", False, f"{type(e).__name__}: {str(e)[:300]}")

print("\nVERIFY_WANDB", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
