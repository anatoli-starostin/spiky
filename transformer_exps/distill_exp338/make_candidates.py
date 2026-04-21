"""Generate candidate configs for the distill exploration sweep."""
import json, os

ROOT = '/home/starost/spiky/transformer_exps/distill_exp338/candidates'

def write(name, luts, d_intermediate, desc):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "name": name, "description": desc,
        "luts": luts, "d_intermediate": d_intermediate,
    }
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def bp(luts):
    # Assume n_heads=1 for all (consistent with out_proj).
    return sum(L['tph'] * (1 << L['input_nap']) * L['output_nap'] for L in luts)


# ---------- Axis A: single-LUT narrow-vs-wide at fixed bit budget ----------
# ~1.3M budget (= c06 / c05 budget)
for in_nap, tph in [(4, 8192), (5, 4096), (6, 2048), (7, 1024), (8, 512), (10, 128), (12, 32)]:
    luts = [{"input_nap": in_nap, "tph": tph, "output_nap": 10}]
    name = f"c_a_in{in_nap:02d}_tph{tph:05d}_on10_bp{bp(luts)//1000:05d}k"
    write(name, luts, [], f"Axis A (1.3M budget): single LUT, in={in_nap}, tph={tph}, on=10")

# ~2.6M budget
for in_nap, tph in [(5, 8192), (6, 4096), (7, 2048), (8, 1024), (10, 256)]:
    luts = [{"input_nap": in_nap, "tph": tph, "output_nap": 10}]
    name = f"c_a_in{in_nap:02d}_tph{tph:05d}_on10_bp{bp(luts)//1000:05d}k"
    write(name, luts, [], f"Axis A (2.6M budget): single LUT, in={in_nap}, tph={tph}, on=10")

# ---------- Axis B: output_nap sweep at (in=6, tph=2048) ----------
for on in [6, 8, 12, 16, 24, 32]:
    luts = [{"input_nap": 6, "tph": 2048, "output_nap": on}]
    name = f"c_b_in06_tph02048_on{on:02d}_bp{bp(luts)//1000:05d}k"
    write(name, luts, [], f"Axis B: in=6, tph=2048, on={on}")

# ---------- Axis C: push c06 direction (narrow + many tables) to lower ----------
for in_nap, tph in [(5, 1024), (6, 1024), (6, 512), (5, 2048), (5, 512), (4, 4096)]:
    luts = [{"input_nap": in_nap, "tph": tph, "output_nap": 10}]
    name = f"c_c_in{in_nap:02d}_tph{tph:05d}_on10_bp{bp(luts)//1000:05d}k"
    write(name, luts, [], f"Axis C (small): single LUT, in={in_nap}, tph={tph}, on=10")

# ---------- Axis D: 2-LUT stacks, various intermediate dims ----------
# 64 -> LUT1(partition) -> D2V(d_mid) -> LUT2 -> 32
D_STACKS = [
    # d_mid=16
    dict(mid=16, L1=dict(input_nap=6, tph=1024, output_nap=16),
         L2=dict(input_nap=6, tph=1024, output_nap=16)),
    dict(mid=16, L1=dict(input_nap=6, tph=512, output_nap=16),
         L2=dict(input_nap=8, tph=512, output_nap=10)),
    dict(mid=16, L1=dict(input_nap=4, tph=2048, output_nap=16),
         L2=dict(input_nap=6, tph=1024, output_nap=10)),
    # d_mid=24
    dict(mid=24, L1=dict(input_nap=6, tph=1024, output_nap=24),
         L2=dict(input_nap=8, tph=512, output_nap=10)),
    dict(mid=24, L1=dict(input_nap=6, tph=512, output_nap=24),
         L2=dict(input_nap=10, tph=256, output_nap=10)),
    # d_mid=8
    dict(mid=8, L1=dict(input_nap=6, tph=1024, output_nap=8),
         L2=dict(input_nap=4, tph=2048, output_nap=10)),
    # d_mid=32 (same as output)
    dict(mid=32, L1=dict(input_nap=6, tph=1024, output_nap=32),
         L2=dict(input_nap=6, tph=512, output_nap=10)),
]
for s in D_STACKS:
    luts = [s['L1'], s['L2']]
    name = ("c_d_mid{mid:02d}_L1in{L1in:02d}tph{L1tph:05d}on{L1on:02d}"
            "_L2in{L2in:02d}tph{L2tph:05d}on{L2on:02d}_bp{bpk:05d}k").format(
        mid=s['mid'], L1in=s['L1']['input_nap'], L1tph=s['L1']['tph'], L1on=s['L1']['output_nap'],
        L2in=s['L2']['input_nap'], L2tph=s['L2']['tph'], L2on=s['L2']['output_nap'],
        bpk=bp(luts)//1000,
    )
    write(name, luts, [s['mid']],
          f"Axis D: 2-LUT stack, d_mid={s['mid']}, L1={s['L1']}, L2={s['L2']}")

# ---------- Axis E: 3-LUT stacks ----------
E_STACKS = [
    dict(mids=[16, 16],
         Ls=[dict(input_nap=6, tph=512, output_nap=16),
             dict(input_nap=6, tph=512, output_nap=16),
             dict(input_nap=6, tph=512, output_nap=10)]),
    dict(mids=[24, 16],
         Ls=[dict(input_nap=6, tph=512, output_nap=24),
             dict(input_nap=8, tph=256, output_nap=16),
             dict(input_nap=6, tph=256, output_nap=10)]),
]
for s in E_STACKS:
    luts = s['Ls']
    name = "c_e_" + "_".join(f"L{i}in{L['input_nap']:02d}tph{L['tph']:05d}on{L['output_nap']:02d}"
                              for i, L in enumerate(luts))
    name += f"_bp{bp(luts)//1000:05d}k"
    write(name, luts, s['mids'], f"Axis E: 3-LUT stack")

# print summary
import glob
cands = sorted(glob.glob(os.path.join(ROOT, 'c_*/config.json')))
print(f'created {len(cands)} candidate configs')
for c in cands:
    cfg = json.load(open(c))
    print(f"  {cfg['name']}  bit_params={bp(cfg['luts']):,}")
