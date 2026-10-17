"""CPU-only: per-layer confidence scalars from saved model checkpoints. Usage: scalars_ckpt.py <ckpt> [<ckpt> ...]"""
import os, sys, math, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

KEYS = ("confidence_log_beta", "confidence_log_gamma", "log_read_tau")
INIT = {"confidence_log_beta": math.log(2.0), "confidence_log_gamma": 0.0, "log_read_tau": math.log(0.5)}

def load(path):
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    out = {}
    for k, v in sd.items():
        for key in KEYS:
            if k.endswith(key):
                layer = int(re.search(r"h\.(\d+)\.", k).group(1))
                out.setdefault(key, {})[layer] = v.float().item()
    return out

runs = [(os.path.basename(p), load(p)) for p in sys.argv[1:]]
for key in KEYS:
    if not any(key in r for _, r in runs):
        print(f"\n{key}: not present in these checkpoints"); continue
    print(f"\n=== {key}  (init log-value {INIT[key]:+.6f}) ===")
    for name, r in runs:
        vals = [r[key][i] for i in sorted(r[key])]
        bit_init = sum(1 for v in vals if v == torch.tensor(INIT[key]).float().item())
        print(f"  {name:22s} n={len(vals)} min {min(vals):+.5f} mean {sum(vals)/len(vals):+.5f} max {max(vals):+.5f} "
              f"| exp(): min {math.exp(min(vals)):.4f} max {math.exp(max(vals)):.4f} | layers bit-equal to init: {bit_init}")
    if len(runs) >= 2:
        (na, a), (nb, b) = runs[0], runs[-1]
        if key in a and key in b:
            d = [abs(b[key][i] - a[key][i]) for i in sorted(a[key])]
            same = sum(1 for x in d if x == 0.0)
            print(f"  delta {na} -> {nb}: max|d| {max(d):.3e} mean|d| {sum(d)/len(d):.3e}  bit-identical layers: {same}/{len(d)}")
    print("  per-layer (last ckpt):", " ".join(f"{runs[-1][1][key][i]:+.4f}" for i in sorted(runs[-1][1][key])))
