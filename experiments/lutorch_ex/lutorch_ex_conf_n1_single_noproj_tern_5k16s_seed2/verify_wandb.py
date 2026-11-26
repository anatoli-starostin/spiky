"""Pull a ternin arm run (env NAME, ARM) from W&B (name from env NAME): val_bpb and val/bpb at steps 500..5000 equal metrics.csv, train/lr follows the 16K schedule, group lutorch_ex, tags.
Usage: NAME=... verify_wandb.py metrics.csv   (as ../../../spiky experiments/lutorch_ex/.../verify_wandb.py of the seed-1 run, without the 16K-original comparison)"""
import csv, os, sys, time, math, wandb
NAME = os.environ["NAME"]; WANT = {"lutorch_ex", "ternin", os.environ["ARM"], "5k", "16k-schedule", "seed2"}
rows = {int(r["step"]): float(r["val_bpb"]) for r in csv.DictReader(open(sys.argv[1]))}
def lr16(s, N=16000, w=1600, peak=3e-4):
    return peak * (s / w if s < w else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - w) / (N - w))))
api = wandb.Api(timeout=60); run = None
for attempt in range(40):
    try:
        run = api.run(f"deepspike-inc/tally_transformer/{NAME}")
        h = {}
        for r in run.scan_history(keys=["_step", "val_bpb", "val/bpb"]):
            if r.get("val_bpb") is not None: h[int(r["_step"])] = (r["val_bpb"], r.get("val/bpb"))
        if max(h, default=0) >= 5000: break
    except Exception as e: print("retry", e)
    time.sleep(30)
else:
    print("FAIL: the curve is not complete on W&B"); sys.exit(1)
print("url", run.url, "state", run.state, "tags", run.tags, "group", run.group)
lrs = {int(r["_step"]): r["train/lr"] for r in run.scan_history(keys=["_step", "train/lr"]) if r.get("train/lr") is not None}
bad = [s for s in rows if s not in h or abs(h[s][0] - rows[s]) > 1e-4 or h[s][1] is None or abs(h[s][1] - rows[s]) > 1e-4]
print("step local wandb_val_bpb wandb_val/bpb")
for s in sorted(rows): print(s, rows[s], h.get(s))
for s in (500, 1000, 1600, 5000):
    w = lrs.get(s); print(f"lr@{s}: wandb={w} expected16K={lr16(s):.6e}")
    if w is None or abs(w - lr16(s)) > 1e-9 + 1e-3 * lr16(s): bad.append(f"lr{s}")
if not WANT <= set(run.tags) or run.group != "lutorch_ex": bad.append(("tags/group", run.tags, run.group))
if bad or sorted(rows) != list(range(500, 5001, 500)): print("MISMATCH", bad); sys.exit(1)
print("WANDB OK")
