"""Pull the run from W&B: check val_bpb and val/bpb (steps 500..5000) match metrics.csv, check train/lr follows the
16K schedule (warm-up 1600, cosine to 0.1x over 16000) and compare the curve with the original 16K run qwnyrzbh."""
import csv, sys, time, math, wandb
NAME = "lutorch_ex_conf_n1_single_noproj_5k16s"
ORIG = {500: 2.0125, 1000: 1.7528, 2000: 1.5093, 3000: 1.3871, 4000: 1.3190, 5000: 1.2790}
rows = {int(r["step"]): float(r["val_bpb"]) for r in csv.DictReader(open(sys.argv[1]))}
def lr16(s, N=16000, w=1600, peak=3e-4):
    return peak * (s / w if s < w else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - w) / (N - w))))
api = wandb.Api(timeout=60)
for attempt in range(20):
    try:
        run = api.run(f"deepspike-inc/tally_transformer/{NAME}")
        h = {}
        for r in run.scan_history(keys=["_step", "val_bpb", "val/bpb"]):
            if r.get("val_bpb") is not None: h[int(r["_step"])] = (r["val_bpb"], r.get("val/bpb"))
        if max(h, default=0) >= 5000: break
    except Exception as e: print("retry", e)
    time.sleep(30)
print("url", run.url, "state", run.state, "tags", run.tags, "group", run.group)
lrs = {int(r["_step"]): r["train/lr"] for r in run.scan_history(keys=["_step", "train/lr"]) if r.get("train/lr") is not None}
bad = [s for s in rows if s not in h or abs(h[s][0] - rows[s]) > 1e-4 or h[s][1] is None or abs(h[s][1] - rows[s]) > 1e-4]
print("step local wandb_val_bpb wandb_val/bpb orig16K diff")
for s in sorted(rows): print(s, rows[s], h.get(s), ORIG.get(s), round(rows[s] - ORIG[s], 4) if s in ORIG else "")
for s in (500, 1000, 1600, 5000):
    w = lrs.get(s); print(f"lr@{s}: wandb={w} expected16K={lr16(s):.6e}")
    if w is None or abs(w - lr16(s)) > 1e-9 + 1e-3 * lr16(s): bad.append(f"lr{s}")
if bad or sorted(rows) != list(range(500, 5001, 500)): print("MISMATCH", bad); sys.exit(1)
print("WANDB OK")
