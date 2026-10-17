"""4-way val bpb / CORE / train-loss comparison at shared steps: dense, v2, confn2 (local logs) + projmuon (worker values)."""
import re
R = "/home/astarostin/projects/lut_nanochat_wt/experiments/lut_nanochat"
LOGS = {
    "dense": f"{R}/results/d24-dense-1xh100-s0/train.log",
    "v2": f"{R}/nanochat/wandb/run-20261014_210912-d24-lutffn-confn1-1xh100-s0-full-v2-9811cced/files/output.log",
    "confn2": f"{R}/results/confn2.nohup.log",
}
B_VAL = {250: 1.050365, 500: 0.960619, 750: 0.921788, 1000: 0.900630, 1250: 0.885783, 1500: 0.876608, 1750: 0.868583, 2000: 0.860700}
B_CORE = {500: 0.0955, 1000: 0.1464, 1500: 0.1531, 2000: 0.1703}
B_LOSS = {250: 3.472070, 500: 3.154972, 750: 3.041262, 1000: 2.951401, 1250: 2.919951, 1500: 2.905614, 1750: 2.862029, 2000: 2.812967}

def parse(path):
    val, core, loss = {}, {}, {}
    for line in open(path, errors="ignore"):
        m = re.match(r"Step (\d+) \| Validation bpb: ([\d.]+)", line)
        if m: val[int(m.group(1))] = float(m.group(2)); continue          # last occurrence wins (resumes)
        m = re.match(r"Step (\d+) \| CORE metric: ([\d.]+)", line)
        if m: core[int(m.group(1))] = float(m.group(2)); continue
        m = re.match(r"step (\d+)/\d+ .*?loss: ([\d.]+)", line)
        if m and int(m.group(1)) % 250 == 0: loss[int(m.group(1))] = float(m.group(2))
    return val, core, loss

D = {k: parse(p) for k, p in LOGS.items()}
D["projmuon"] = (B_VAL, B_CORE, B_LOSS)
names = ["dense", "v2", "confn2", "projmuon"]
for n in names:
    v = D[n][0]
    print(f"{n:9s} val points: {len(v)} (max step {max(v)})")

steps = sorted(s for s in D["v2"][0] if s > 0 and (s in D["confn2"][0] or s in B_VAL))
print("\nVAL BPB (delta vs v2; negative = better than v2)")
print(f"{'step':>5} {'dense':>9} {'v2':>9} {'confn2':>9} {'d(c2-v2)':>9} {'projmuon':>9} {'d(pm-v2)':>9} {'d(v2-dense)':>11}")
for s in steps:
    dn, v2 = D["dense"][0].get(s), D["v2"][0][s]
    c2, pm = D["confn2"][0].get(s), B_VAL.get(s)
    f = lambda x: f"{x:9.6f}" if x is not None else f"{'-':>9}"
    g = lambda x: f"{x - v2:+9.6f}" if x is not None else f"{'-':>9}"
    print(f"{s:>5} {f(dn)} {v2:9.6f} {f(c2)} {g(c2)} {f(pm)} {g(pm)} {v2 - dn:+11.6f}" if dn else
          f"{s:>5} {f(dn)} {v2:9.6f} {f(c2)} {g(c2)} {f(pm)} {g(pm)} {'-':>11}")

print("\nCORE (in-training 500-ex subsample, noise ~0.008)")
for s in sorted(D["v2"][1]):
    if s in D["confn2"][1] or s in B_CORE:
        v2 = D["v2"][1][s]
        c2, pm, dn = D["confn2"][1].get(s), B_CORE.get(s), D["dense"][1].get(s)
        print(f"  {s:>5} dense {dn if dn is not None else '-'}  v2 {v2:.4f}  confn2 {c2 if c2 is not None else '-'} "
              f"({(c2 - v2):+.4f})" if c2 is not None else f"  {s:>5} dense {dn}  v2 {v2:.4f}  confn2 -", end="")
        print(f"  projmuon {pm:.4f} ({pm - v2:+.4f})" if pm is not None else "  projmuon -")

print("\nTRAIN LOSS delta vs v2 (EMA-smoothed, single step)")
for s in steps:
    v2 = D["v2"][2].get(s); c2 = D["confn2"][2].get(s); pm = B_LOSS.get(s)
    print(f"  {s:>5} v2 {v2:.4f}  confn2 {('%+.4f' % (c2 - v2)) if c2 else '-'}  projmuon {('%+.4f' % (pm - v2)) if pm else '-'}")

print("\nDENSE final:", {k: D['dense'][0][k] for k in sorted(D['dense'][0])[-2:]}, "| v2 final:", {k: D['v2'][0][k] for k in sorted(D['v2'][0])[-2:]})
