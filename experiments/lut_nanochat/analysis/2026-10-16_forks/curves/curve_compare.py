"""READ-ONLY curve + geometry comparison for BODY_TASK c98db93d.
Pull loss/val-bpb series and config for the live fork, v2, and the aborted run from wandb.
Align by step; print numeric diff tables. Touches nothing. Runs OUT of cage (network)."""
import math
import wandb

ENT, PROJ = "anatoli-starostin-relocation", "spiky-nanochat"
api = wandb.Api()
runs = list(api.runs(f"{ENT}/{PROJ}"))

def find(suf):
    ms = [r for r in runs if suf in r.id]
    return ms[0] if ms else None

live = find("7f4cc747")   # fork r1536
v2   = find("9811cced")   # v2 r768
ab   = find("3f8b4134")   # aborted fork (5280 horizon)

print("=== RUNS ===")
for tag, r in [("LIVE/fork", live), ("v2", v2), ("aborted", ab)]:
    print(f"{tag:10s}: {'NOT FOUND' if r is None else r.id + ' | state=' + str(r.state)}")

CFG_KEYS = ["depth","n_embd","num_iterations","lut_ffn","lut_ffn_h","lut_ffn_tph","lut_ffn_d",
            "lut_ffn_nap","lut_ffn_read_top_n","lut_ffn_proj_bf16","lut_ffn_proj_fp8_muon",
            "no_lr_rescale","fp8","device_batch_size","total_batch_size","target_param_data_ratio"]

def dump_cfg(tag, r):
    if r is None:
        print(f"\n--- {tag} config: run missing ---"); return
    c = {k: r.config.get(k) for k in CFG_KEYS if k in r.config}
    print(f"\n--- {tag} config (selected) ---")
    for k in CFG_KEYS:
        if k in r.config:
            print(f"  {k} = {r.config[k]}")
    # param-count-ish keys from config + summary
    extra = {k: v for k, v in r.config.items() if ("param" in k.lower() or "lut" in k.lower()) and k not in c}
    if extra:
        print("  [other param/lut config keys]")
        for k, v in extra.items():
            print(f"    {k} = {v}")
    sk = {k: v for k, v in dict(r.summary).items() if any(t in k.lower() for t in ("param","lut","embd","depth"))}
    if sk:
        print("  [param-ish summary keys]")
        for k, v in sk.items():
            print(f"    {k} = {v}")

for tag, r in [("LIVE/fork", live), ("v2", v2), ("aborted", ab)]:
    dump_cfg(tag, r)

# discover history columns from live
print("\n=== LIVE summary keys ===")
print(sorted(dict(live.summary).keys()))

def series(r, prefer):
    """Return {step: value} for the first key that exists among `prefer`, using scan_history."""
    if r is None:
        return {}, None
    cols = None
    for key in prefer:
        try:
            hist = list(r.scan_history(keys=["_step", key]))
        except Exception:
            hist = []
        d = {row["_step"]: row[key] for row in hist if row.get(key) is not None and row.get("_step") is not None}
        if d:
            return d, key
    return {}, None

LOSS_KEYS = ["train/loss","train_loss","loss","lossf","train/lossf","step_loss"]
BPB_KEYS  = ["val/bpb","val_bpb","bpb","eval/val_bpb","Validation bpb","val/val_bpb"]

live_loss, lk = series(live, LOSS_KEYS)
v2_loss, lk2 = series(v2, LOSS_KEYS)
ab_loss, lk3 = series(ab, LOSS_KEYS)
print(f"\n=== loss key used: live={lk} v2={lk2} aborted={lk3} ===")
print(f"n points: live={len(live_loss)} v2={len(v2_loss)} aborted={len(ab_loss)}")

def fmt(v):
    return f"{v:.6f}" if isinstance(v, (int, float)) else str(v)

# ---- aborted vs live (expect identical early) ----
print("\n=== ABORTED vs LIVE  (train loss; same geometry+seed, horizons 5280 vs 5568) ===")
print(f"{'step':>6} {'aborted':>12} {'live':>12} {'abs_diff':>12}")
for s in sorted(set(ab_loss) & set(live_loss)):
    a, b = ab_loss[s], live_loss[s]
    print(f"{s:>6} {fmt(a):>12} {fmt(b):>12} {abs(a-b):>12.2e}")

# ---- v2 vs live (the suspicious one) ----
print("\n=== v2 vs LIVE  (train loss; DIFFERENT geometry r768 vs r1536) ===")
checkpoints = [0,1,2,3,4,5,8,10,15,20,30,50,75,100,150,200,300,400,500,750,1000,1250,1500]
common = sorted(set(v2_loss) & set(live_loss))
# use checkpoints that exist in both, plus nearest available
print(f"overlap steps: {len(common)} (min {min(common) if common else '-'}, max {max(common) if common else '-'})")
print(f"{'step':>6} {'v2':>12} {'live':>12} {'live-v2':>12} {'rel%':>8}")
rows = [s for s in checkpoints if s in v2_loss and s in live_loss]
for s in rows:
    a, b = v2_loss[s], live_loss[s]
    rel = (b-a)/a*100 if a else float('nan')
    print(f"{s:>6} {fmt(a):>12} {fmt(b):>12} {b-a:>+12.6f} {rel:>+8.3f}")

# gap trend summary
if common:
    def gap_at(s):
        return live_loss[s]-v2_loss[s]
    early = [gap_at(s) for s in common if s <= 10]
    mid   = [gap_at(s) for s in common if 100 <= s <= 300]
    late  = [gap_at(s) for s in common if s >= 800]
    def mean(x): return sum(x)/len(x) if x else float('nan')
    print(f"\nmean(live-v2) gap: early(<=10)={mean(early):+.4f}  mid(100-300)={mean(mid):+.4f}  late(>=800)={mean(late):+.4f}")

# ---- val bpb ----
live_bpb, bk = series(live, BPB_KEYS)
v2_bpb, bk2 = series(v2, BPB_KEYS)
print(f"\n=== val bpb key used: live={bk} v2={bk2} ===")
print(f"{'step':>6} {'v2_bpb':>12} {'live_bpb':>12} {'live-v2':>12}")
for s in sorted(set(v2_bpb) & set(live_bpb)):
    a, b = v2_bpb[s], live_bpb[s]
    print(f"{s:>6} {fmt(a):>12} {fmt(b):>12} {b-a:>+12.6f}")

# ---- cost / param evidence from wandb summary/config ----
print("\n=== COST / PARAM evidence (wandb) ===")
for tag, r in [("LIVE/fork", live), ("v2", v2)]:
    if r is None: continue
    s = dict(r.summary)
    interesting = {k: v for k, v in s.items() if any(t in k.lower() for t in ("param","runtime","dt","tok","mfu","mem","step"))}
    print(f"[{tag}] runtime={r.summary.get('_runtime')} step={r.summary.get('_step')}")
    for k in sorted(interesting):
        print(f"    {k} = {interesting[k]}")
print("\nln(32768) =", math.log(32768))
print("COMPARE_DONE")
