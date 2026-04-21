"""Generate layer-0-focused candidates. Layer 0 was hardest to distill
(0.887 at 1.3M), so this sweep pushes budgets higher and explores shapes
that benefit the harder mapping.
"""
import json, os

ROOT = '/home/starost/spiky/transformer_exps/distill_exp338/candidates_l0'
os.makedirs(ROOT, exist_ok=True)

def write(name, luts, d_intermediate, desc):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump({"name": name, "description": desc, "luts": luts,
                   "d_intermediate": d_intermediate}, f, indent=2)

def bp(luts):
    return sum(L['tph'] * (1 << L['input_nap']) * L['output_nap'] for L in luts)


# ---------- Single-LUT scans at larger budgets ----------

# Narrow × many at 2.6M / 5.2M / 10.5M
for in_nap, tph in [(4, 16384), (5, 8192), (6, 4096)]:
    luts = [{"input_nap": in_nap, "tph": tph, "output_nap": 10}]
    write(f"l0_s_in{in_nap:02d}_tph{tph:05d}_on10_{bp(luts)//1000:05d}k",
          luts, [], f"2.6M: in={in_nap}, tph={tph}, on=10")

for in_nap, tph in [(4, 32768), (5, 16384), (6, 8192), (7, 4096), (8, 2048)]:
    luts = [{"input_nap": in_nap, "tph": tph, "output_nap": 10}]
    write(f"l0_s_in{in_nap:02d}_tph{tph:05d}_on10_{bp(luts)//1000:05d}k",
          luts, [], f"5.2M: in={in_nap}, tph={tph}, on=10")

for in_nap, tph in [(6, 16384), (7, 8192), (8, 4096), (10, 1024)]:
    luts = [{"input_nap": in_nap, "tph": tph, "output_nap": 10}]
    write(f"l0_s_in{in_nap:02d}_tph{tph:05d}_on10_{bp(luts)//1000:05d}k",
          luts, [], f"10.5M: in={in_nap}, tph={tph}, on=10")

# Teacher-shape ceiling (21M). We'll also run with load_teacher_pairs via runner flag.
luts = [{"input_nap": 10, "tph": 2048, "output_nap": 10}]
write(f"l0_s_teacher_shape", luts, [], "21M: teacher shape (in=10, tph=2048, on=10)")

# ---------- Output_nap scans at best-looking shape ----------
for tph, on in [(4096, 16), (4096, 24), (4096, 32)]:
    luts = [{"input_nap": 6, "tph": tph, "output_nap": on}]
    write(f"l0_b_in06_tph{tph:05d}_on{on:02d}_{bp(luts)//1000:05d}k",
          luts, [], f"output_nap scan at (in=6, tph={tph}): on={on}")

for tph in [2048, 4096]:
    on = 32
    luts = [{"input_nap": 5, "tph": tph, "output_nap": on}]
    write(f"l0_b_in05_tph{tph:05d}_on{on:02d}_{bp(luts)//1000:05d}k",
          luts, [], f"narrow+high-on: in=5, tph={tph}, on={on}")

# ---------- Two-LUT stacks at meaningful budgets ----------
# Each stack: 64 -> LUT1(partition, D2V after) -> LUT2 -> 32.
D_STACKS = [
    # d_mid=16, ~2.6M
    dict(mid=16, L1=dict(input_nap=6, tph=2048, output_nap=16),
         L2=dict(input_nap=6, tph=1024, output_nap=10)),
    # d_mid=32, ~5M
    dict(mid=32, L1=dict(input_nap=6, tph=2048, output_nap=32),
         L2=dict(input_nap=6, tph=1024, output_nap=10)),
    # d_mid=24, ~5M
    dict(mid=24, L1=dict(input_nap=6, tph=2048, output_nap=24),
         L2=dict(input_nap=8, tph=1024, output_nap=10)),
    # d_mid=16 bigger, ~5M
    dict(mid=16, L1=dict(input_nap=6, tph=4096, output_nap=16),
         L2=dict(input_nap=8, tph=1024, output_nap=10)),
    # narrow+deep: d_mid=16 + tiny L1 + fat L2
    dict(mid=16, L1=dict(input_nap=4, tph=4096, output_nap=16),
         L2=dict(input_nap=8, tph=2048, output_nap=10)),
    # large stack ~10M
    dict(mid=32, L1=dict(input_nap=6, tph=4096, output_nap=32),
         L2=dict(input_nap=8, tph=1024, output_nap=10)),
]
for s in D_STACKS:
    luts = [s['L1'], s['L2']]
    name = (f"l0_d_mid{s['mid']:02d}"
            f"_L1in{s['L1']['input_nap']:02d}tph{s['L1']['tph']:05d}on{s['L1']['output_nap']:02d}"
            f"_L2in{s['L2']['input_nap']:02d}tph{s['L2']['tph']:05d}on{s['L2']['output_nap']:02d}"
            f"_{bp(luts)//1000:05d}k")
    write(name, luts, [s['mid']], f"2-LUT stack mid={s['mid']}")

# ---------- 3-LUT stack ----------
E_STACKS = [
    dict(mids=[16, 16],
         Ls=[dict(input_nap=6, tph=1024, output_nap=16),
             dict(input_nap=6, tph=1024, output_nap=16),
             dict(input_nap=6, tph=1024, output_nap=10)]),
    dict(mids=[32, 16],
         Ls=[dict(input_nap=6, tph=1024, output_nap=32),
             dict(input_nap=6, tph=1024, output_nap=16),
             dict(input_nap=6, tph=1024, output_nap=10)]),
]
for s in E_STACKS:
    luts = s['Ls']
    name = "l0_e_" + "_".join(
        f"L{i}in{L['input_nap']:02d}tph{L['tph']:05d}on{L['output_nap']:02d}"
        for i, L in enumerate(luts)
    ) + f"_{bp(luts)//1000:05d}k"
    write(name, luts, s['mids'], f"3-LUT stack")

import glob
cands = sorted(glob.glob(os.path.join(ROOT, '*/config.json')))
print(f'created {len(cands)} layer-0 candidates')
for c in cands:
    cfg = json.load(open(c))
    print(f"  {cfg['name']:<70}  bp={bp(cfg['luts'])//1000:>6}k")
