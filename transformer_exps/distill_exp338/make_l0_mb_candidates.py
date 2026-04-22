"""Generate MultiBit K=4 candidates for layer-0 distillation topology sweep."""
import json, os

ROOT = '/home/starost/spiky/transformer_exps/distill_exp338/candidates_l0_mb'
os.makedirs(ROOT, exist_ok=True)

def write(name, luts, desc):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "name": name,
        "module_type": "multibit",
        "bit_width": 4,
        "init_std": 0.001,
        "description": desc,
        "luts": luts,
        "d_intermediate": [],
    }
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

def bp(luts):
    return sum(L['tph'] * (1 << L['input_nap']) * L['output_nap'] for L in luts)


# Axis 1: input_nap sweep at (tph=256, on=32) — 7 candidates
for in_nap in (4, 5, 6, 7, 8, 10, 12):
    luts = [{"input_nap": in_nap, "tph": 256, "output_nap": 32}]
    name = f"mb_a1_in{in_nap:02d}_tph00256_on32_{bp(luts)//1000:06d}k"
    write(name, luts, f"Axis 1: in={in_nap}, tph=256, on=32, K=4")

# Axis 2: input_nap at tph=512 — 4 candidates
for in_nap in (4, 6, 8, 10):
    luts = [{"input_nap": in_nap, "tph": 512, "output_nap": 32}]
    name = f"mb_a2_in{in_nap:02d}_tph00512_on32_{bp(luts)//1000:06d}k"
    write(name, luts, f"Axis 2: in={in_nap}, tph=512, on=32, K=4")

# Axis 3: tph sweep at (in=10, on=32) — 4 candidates (256 in axis 1)
for tph in (64, 128, 512, 1024):
    luts = [{"input_nap": 10, "tph": tph, "output_nap": 32}]
    name = f"mb_a3_in10_tph{tph:05d}_on32_{bp(luts)//1000:06d}k"
    write(name, luts, f"Axis 3: in=10, tph={tph}, on=32, K=4")

# Axis 4: output_nap at (in=10, tph=256) — 3 candidates (32 in axis 1)
for on in (10, 16, 48):
    luts = [{"input_nap": 10, "tph": 256, "output_nap": on}]
    name = f"mb_a4_in10_tph00256_on{on:02d}_{bp(luts)//1000:06d}k"
    write(name, luts, f"Axis 4: in=10, tph=256, on={on}, K=4")

# Print summary
import glob
cands = sorted(glob.glob(os.path.join(ROOT, 'mb_a*/config.json')))
print(f'created {len(cands)} MultiBit K=4 candidates')
for c in cands:
    cfg = json.load(open(c))
    print(f"  {cfg['name']:<50}  bp={bp(cfg['luts'])//1000:>6}k  ({bp(cfg['luts'])//1000 * 4 // 1000}Mbit)")
