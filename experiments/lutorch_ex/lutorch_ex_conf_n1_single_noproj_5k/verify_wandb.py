"""Pull the run from W&B and check the val_bpb history (steps 500..5000) matches metrics.csv; print the curve."""
import csv, sys, time, os, wandb
rows = {int(r["step"]): float(r["val_bpb"]) for r in csv.DictReader(open(sys.argv[1]))}
api = wandb.Api(timeout=60)
for attempt in range(10):
    try:
        run = api.run("deepspike-inc/tally_transformer/lutorch_ex_conf_n1_single_noproj_5k")
        hist = {int(r["_step"]): r["val_bpb"] for r in run.scan_history(keys=["_step", "val_bpb"]) if r.get("val_bpb") is not None}
        if max(hist, default=0) >= 5000: break
    except Exception as e: print("retry", e)
    time.sleep(30)
print("url", run.url, "state", run.state, "tags", run.tags, "group", run.group)
bad = [s for s in rows if s not in hist or abs(hist[s] - rows[s]) > 1e-4]
for s in sorted(rows): print(s, rows[s], hist.get(s))
if bad or sorted(rows) != list(range(500, 5001, 500)): print("MISMATCH", bad); sys.exit(1)
print("WANDB OK")
