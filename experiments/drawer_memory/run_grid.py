"""Run a list of configs sequentially, cooling the M1 between runs. Usage: run_grid.py <grid_name>

Per run: runs/<name>/{config.json, metrics.csv, summary.json, thermal.jsonl, train.log}
After every run: runs/results.csv (all finished runs) and runs/PROGRESS.md (human-readable status).
"""
import csv
import json
import os
import subprocess
import sys
import time

from thermal import Thermal

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PY = sys.executable
COOL_TO = 65.0
NAP_FOR = {64: 4, 256: 6}  # memory per head F = 4 tables * 2**nap

GRIDS = {
    "pilot": [dict(mixer=m, n_pairs=16, feat=256) for m in ["softmax", "linear", "lut_hard", "lut_nc"]],
    # change after pilot: lut_* sat at chance; tie Q/K projection (Reformer's requirement for hashed attention)
    "tied": [dict(mixer=m, n_pairs=16, feat=256) for m in ["lut_hard_tied", "lut_nc_tied"]],
}

# night grid, 2026-10-18. A: drawers vs linear at equal memory under pressure (2 seeds);
# B: which spiky trick slows learning; C: 64 pairs if time remains.
# ordered by priority: the decisive 32-pair comparison first, in case thermal pauses eat the night
A = lambda n: [dict(block="A", mixer=m, n_pairs=n, feat=F, seed=seed)
               for F in (64, 256) for seed in (0, 1) for m in ("linear", "lut_hard_tied")]
night = A(32)
night += [dict(block="A", mixer="softmax", n_pairs=32, feat=256, seed=s) for s in (0, 1)]
night += [dict(block="B", mixer=m, n_pairs=16, feat=256, seed=0) for m in ("lut_nbr_tied", "lut_conf_tied")]
night += [dict(block="B", mixer="lut_nc_tied", n_pairs=16, feat=256, seed=0, steps=6000)]
night += A(16)
night += [dict(block="C", mixer=m, n_pairs=64, feat=64, seed=0) for m in ("linear", "lut_hard_tied")]
GRIDS["night"] = night

# night2, 2026-10-19 (after night: drawers click later than linear and sometimes not within 3000 rounds;
# neighbour drawer helped, confidence hurt). D: neighbour drawers on hard settings; E: 4 questions per
# cabinet with 4x more cabinets (same memory); F: 10k rounds for runs that never clicked (+ linear for fairness).
night2 = [dict(block="D", mixer="lut_nbr_tied", n_pairs=32, feat=F, seed=s) for F in (64, 256) for s in (0, 1)]
night2 += [dict(block="D", mixer="lut_nbr_tied", n_pairs=64, feat=64, seed=0)]
night2 += [dict(block="E", mixer=m, n_pairs=32, feat=256, nap=4, seed=s)
           for m in ("lut_hard_tied", "lut_nbr_tied") for s in (0, 1)]
night2 += [dict(block="F", mixer="lut_hard_tied", n_pairs=32, feat=256, seed=0, steps=10000)]
night2 += [dict(block="F", mixer=m, n_pairs=64, feat=64, seed=0, steps=10000)
           for m in ("linear", "lut_hard_tied", "lut_nbr_tied")]
GRIDS["night2"] = night2

# poc, 2026-10-19: L = char-level LM on Tiny Shakespeare (4 layers, T=256, shift over 4 previous chars);
# M = MQAR under pressure with anchors drawn per seed: gaps between key and value, 64 pairs with small tables.
poc = [dict(block="L", script="train_lm.py", mixer=m, feat=256, nap=4, seed=s)
       for s in (0, 1) for m in ("softmax", "linear", "lut_hard_tied", "lut_nbr_tied")]
poc += [dict(block="M", mixer=m, n_pairs=32, feat=256, nap=4, gap=2, shift_k=4, anchor_per_seed=True, seed=s)
        for s in (0, 1) for m in ("linear", "lut_hard_tied")]
poc += [dict(block="M", mixer="softmax", n_pairs=32, feat=256, gap=2, shift_k=4, anchor_per_seed=True, seed=0)]
poc += [dict(block="M", mixer=m, n_pairs=64, feat=256, nap=4, anchor_per_seed=True, seed=0)
        for m in ("linear", "lut_hard_tied")]
poc += [dict(block="M", mixer="lut_hard_tied", n_pairs=64, feat=256, nap=4, anchor_per_seed=True, seed=1)]
poc += [dict(block="M", mixer="lut_hard_tied", n_pairs=64, feat=64, nap=2, anchor_per_seed=True, seed=0)]
GRIDS["poc"] = poc

# next (batch 6), after external review 2026-10-19. All MQAR runs draw anchors per seed.
AP = dict(anchor_per_seed=True)
nxt = []
# D: address diagnostics over training (checkpoint every 250 steps), 32 pairs, F=256
nxt += [dict(block="D", mixer=m, n_pairs=32, feat=256, nap=b, seed=0, ckpt_every=250, **AP)
        for m, b in (("lut_hard_tied", 4), ("lut_hard_tied", 6), ("lut_nbr_tied", 6))]
# P: parameter-matched linear attention (F=32: ~same parameters as drawers, 8x smaller state)
nxt += [dict(block="P", mixer="linear", n_pairs=32, feat=32, seed=s, **AP) for s in (0, 1)]
# U: memory editing -- 16 pairs, 8 of them re-assigned before the queries
nxt += [dict(block="U", mixer=m, n_pairs=16, updates=8, feat=256, seed=0, **AP) for m in ("softmax", "linear")]
nxt += [dict(block="U", mixer="lut_hard_tied", n_pairs=16, updates=8, feat=256, nap=4, write=wr, seed=s, **AP)
        for wr in ("add", "decay", "ema") for s in (0, 1)]
# X: accuracy vs cells touched at fixed F=256 -- 64 tables x 2 bits; soft intersection (read_power 2) on 16 x 4
nxt += [dict(block="X", mixer="lut_hard_tied", n_pairs=32, feat=256, nap=2, seed=s, **AP) for s in (0, 1)]
nxt += [dict(block="X", mixer="lut_hard_tied", n_pairs=32, feat=256, nap=4, read_power=2, seed=s, **AP) for s in (0, 1)]
# H: local window (16) + drawers hybrid, on recall with gaps and on Shakespeare
nxt += [dict(block="H", mixer=m, n_pairs=32, feat=256, nap=4, gap=2, shift_k=4, seed=0, **AP) for m in ("hybrid", "local")]
nxt += [dict(block="H", script="train_lm.py", mixer=m, feat=256, nap=4, seed=s) for m in ("hybrid", "local") for s in (0, 1)]
nxt += [dict(block="P", script="train_lm.py", mixer="linear", feat=32, seed=0)]
GRIDS["next"] = nxt

# b7 (batch 7): fair softmax reference (RoPE), soft-to-hard address annealing, accuracy-vs-cost curve,
# learning-rate check, and one scale-up step on Shakespeare (d=256, 4 layers, T=512).
GAP = dict(n_pairs=32, feat=256, gap=2, shift_k=4, **AP)
b7 = [dict(block="R", mixer="softmax_rope", seed=s, **GAP) for s in (0, 1)]
b7 += [dict(block="R", script="train_lm.py", mixer="softmax_rope", feat=256, seed=s) for s in (0, 1)]
b7 += [dict(block="A", mixer="lut_hard_tied", nap=4, anneal_frac=0.3, seed=s, **GAP) for s in (0, 1)]
b7 += [dict(block="A", mixer="lut_hard_tied", n_pairs=64, feat=256, nap=4, anneal_frac=0.3, seed=0, **AP)]
b7 += [dict(block="C", mixer="linear", n_pairs=32, feat=f, seed=0, **AP) for f in (64, 128)]
b7 += [dict(block="C", mixer="lut_hard_tied", n_pairs=32, feat=f, nap=4, seed=0, **AP) for f in (64, 128)]
b7 += [dict(block="C", mixer="lut_hard_tied", n_pairs=32, feat=256, nap=4, seed=0, **AP)]
b7 += [dict(block="L", mixer=m, nap=4, lr=lr, seed=0, **GAP) for m in ("linear", "lut_hard_tied") for lr in (3e-4, 3e-3)]
SCALE = dict(script="train_lm.py", feat=256, nap=4, d=256, layers=4, T=512, bs=16, seed=0)
b7 += [dict(block="S", mixer=m, **SCALE) for m in ("softmax_rope", "linear", "lut_hard_tied")]
# H2 (added after the half-width hybrid failed on gaps): layer 1 = local softmax window 16, layer 2 = full drawers
b7 = [dict(block="H2", mixer="local_then_lut_hard_tied", nap=4, seed=s, **GAP) for s in (0, 1)] + b7
GRIDS["b7"] = b7

# pilot8 (2026-10-20): go/no-go for an 8-hour scale-up on enwik8 (byte-level, 90M training bytes).
# Pre-registered rule: go if drawers are within 0.05 bpc of linear, both still improving at the end, and the
# train/held-out gap is < 0.05 bpc; no-go if drawers trail by > 0.10; in between: one more cheap check.
E8 = dict(script="train_lm.py", corpus="enwik8", feat=256, nap=4, steps=1500, seed=0)
GRIDS["pilot8"] = [dict(block="E", mixer="linear", **E8), dict(block="E", mixer="lut_hard_tied", anneal_frac=0.3, **E8)]
GRIDS["pilot8"] += [dict(block="E", mixer="lut_hard_tied", anneal_frac=0.6, **E8)]  # in-between zone: longer soft phase
# pilot8b: bounded soft read -- top-4 cells per table weighted by their probability (user's idea, 2026-10-20)
GRIDS["pilot8b"] = [dict(block="E2", mixer="lut_hard_tied", topk=4, **dict(E8, nap=b)) for b in (4, 6)]
GRIDS["pilot8c"] = [dict(block="E3", mixer="lut_hard_tied", bitflip=2, **dict(E8, nap=6))]  # cheap top-4: flip 2 bits
GRIDS["pilot8"] = GRIDS["pilot8"] + GRIDS["pilot8b"] + GRIDS["pilot8c"]  # one report section for the enwik8 pilot
# e8confirm (2026-10-20, before posting the issue): second seeds of the three enwik8 headline models, and one
# 4,000-step pair (linear vs cheap soft-read drawers) to see whether the gap shrinks or grows with training
E8b = dict(E8, seed=1)
GRIDS["e8confirm"] = [dict(block="E4", mixer="linear", **E8b),
                      dict(block="E4", mixer="lut_hard_tied", anneal_frac=0.3, **E8b),
                      dict(block="E4", mixer="lut_hard_tied", bitflip=2, **dict(E8b, nap=6))]
GRIDS["e8confirm"] += [dict(block="E5", mixer="linear", **dict(E8, steps=4000)),
                       dict(block="E5", mixer="lut_hard_tied", bitflip=2, **dict(E8, nap=6, steps=4000))]
GRIDS["pilot8"] = GRIDS["pilot8"] + GRIDS["e8confirm"]
# augment (2026-10-20): ideas from Sparse Delta Memory on the setting where the text gap shows (enwik8, 4,000 steps,
# 4 x 6-bit tables). Baselines: flip-2 drawers 2.150 bpc, linear attention 2.057.
A4K = dict(script="train_lm.py", corpus="enwik8", feat=256, nap=6, steps=4000, seed=0, block="A2")
GRIDS["augment"] = [dict(mixer="lut_hard_tied", bitflip=4, **A4K),
                    dict(mixer="lut_hard_tied", bitflip=2, init_mem=True, **A4K),
                    dict(mixer="lut_hard_tied", bitflip=2, out_gate=True, **A4K),
                    dict(mixer="lut_hard_tied", bitflip=4, init_mem=True, out_gate=True, **A4K)]
GRIDS["pilot8"] = GRIDS["pilot8"] + GRIDS["augment"]
# scalecheck (2026-10-20): will it fly at scale? (1) fixed vs learned addressing, everything else identical
# (4 x 64-cell tables, 4 cells per token, softmax-weighted); (2) gap vs model size, linear vs flip-2 drawers.
# revised 21:20 after the gated read beat linear attention: fair control first (same gate on linear attention),
# a second seed, then addressing and scaling with the gate on every model.
GRIDS["scalecheck"] = [dict(mixer="linear", out_gate=True, **dict(A4K, block="G")),
                       # seed 1 dropped 23:25 (user: gate verdict is clear)
                       dict(mixer="lut_hard_tied", learned_addr=4, out_gate=True, **dict(A4K, block="L"))]
GRIDS["scalecheck"] += [dict(block="S2", script="train_lm.py", corpus="enwik8", feat=256, nap=6, steps=4000, seed=0,
                             d=dd, layers=4, T=256, mixer=m, out_gate=True, **({"bitflip": 2} if m != "linear" else {}))
                        for dd in (192, 256) for m in ("linear", "lut_hard_tied")]
# structural (2026-10-20 23:40): techniques native to drawers, all with the gated read and flip-2 (16 cells touched per
# head, equal decode cost); compare with gated linear attention 2.005 / gated drawers 2.051.
# (1) bigger tables at the same cost: 4 x 8 bits (memory 1024); (2) Count-Sketch signed writes; (3) key-checked reads.
GRIDS["structural"] = [dict(mixer="lut_hard_tied", bitflip=2, out_gate=True, **dict(A4K, block="X", feat=1024, nap=8)),
                       dict(mixer="lut_hard_tied", bitflip=2, out_gate=True, signed=True, **dict(A4K, block="X")),
                       dict(mixer="lut_hard_tied", bitflip=2, out_gate=True, keycheck=True, **dict(A4K, block="X"))]
# delta (2026-10-21 08:00): delta writes (read, then correct the cell towards the value, learned write strength)
# on recall with 8 of 16 keys re-assigned -- additive memories stall at ~98% on updated keys. Matched pairs:
# additive vs delta, drawers (4 x 6 bits, flip-2) vs linear attention (DeltaNet); all with the gated read.
D = dict(block="D", n_pairs=16, feat=256, nap=6, updates=8, anchor_per_seed=True, out_gate=True, seed=0)
GRIDS["delta"] = [dict(mixer="lut_delta", bitflip=2, **D), dict(mixer="linear_delta", **D)]
# additive gated references dropped 10:20 (user: ~98% ceiling already known). enwik8: delta drawers vs gated DeltaNet
GRIDS["delta_lm"] = [dict(mixer="lut_delta", bitflip=2, out_gate=True, **dict(A4K, block="D2")),
                     dict(mixer="linear_delta", out_gate=True, **dict(A4K, block="D2"))]
GRIDS["pilot8"] = GRIDS["pilot8"] + GRIDS["scalecheck"] + GRIDS["structural"]

# capacity (2026-10-21 11:00): at EQUAL COST PER TOKEN, can a big sparse memory store more associations than a
# dense one? Delta drawers touch 16 cells per head (4 tables, flip-2) whatever the memory size; DeltaNet with a
# feat-wide state touches feat rows. Equal cost = DeltaNet feat 16; feat 64 / 256 are 4x / 16x the cost.
# Plain recall, n pairs (sequence 4n). cap64 runs on the M1, cap128 on the M4.
def _cap(n):
    C = dict(block="C", n_pairs=n, out_gate=True, anchor_per_seed=True, seed=0, steps=3000)
    return [dict(mixer="lut_delta", feat=4096, nap=10, bitflip=2, **C), dict(mixer="linear_delta", feat=16, **C),
            dict(mixer="linear_delta", feat=256, **C), dict(mixer="lut_delta", feat=256, nap=6, bitflip=2, **C),
            dict(mixer="linear_delta", feat=64, **C), dict(mixer="lut_delta", feat=1024, nap=8, bitflip=2, **C)]
GRIDS["cap64"], GRIDS["cap128"] = _cap(64), _cap(128)
# capmini (2026-10-21 12:35): small, fast early-signal version -- 32 pairs (sequence 128), 1,500 steps; the larger
# grids are paused (resumable) and continue only if drawers show an advantage here.
_M = dict(block="M", n_pairs=32, out_gate=True, anchor_per_seed=True, seed=0, steps=1500)
# 13:25: at 1,500 steps even DeltaNet 256 stayed at chance (0.8%) -> too short to tell anything; v2 = 3,000 steps
_M2 = dict(_M, steps=3000)
GRIDS["capmini2_m4"] = [dict(mixer="lut_delta", feat=4096, nap=10, bitflip=2, anneal_frac=0.5, **_M2),
                        dict(mixer="linear_delta", feat=256, **_M2)]
GRIDS["capmini2_m1"] = [dict(mixer="linear_delta", feat=16, **_M2), dict(mixer="lut_delta", feat=256, nap=6, bitflip=2, **_M2)]
GRIDS["capmini_a"] = [dict(mixer="lut_delta", feat=1024, nap=8, bitflip=2, **_M), dict(mixer="linear_delta", feat=16, **_M)]
# capmini_c (13:15): 4,096 cells never left chance in 1,500 steps (no chance matches -> no signal). Coarse-to-fine bit
# curriculum: 4 active bits per table at the start, all 10 by 50% of training.
GRIDS["capmini_c"] = [dict(mixer="lut_delta", feat=4096, nap=10, bitflip=2, anneal_frac=0.5, **_M)]   # M4
GRIDS["capmini_d"] = [dict(mixer="lut_delta", feat=256, nap=6, bitflip=2, **_M)]                        # M1
GRIDS["capmini_b"] = [dict(mixer="lut_delta", feat=4096, nap=10, bitflip=2, **_M), dict(mixer="linear_delta", feat=256, **_M)]


def nap_of(c):
    return c.get("nap", NAP_FOR.get(c["feat"], 6))


def run_name(c):
    if c.get("block") is None:  # legacy names (pilot / tied)
        return f"{c['mixer']}_n{c['n_pairs']}_F{c['feat']}_s{c.get('seed', 0)}"
    extra = (f"_learned{c['learned_addr']}" if c.get("learned_addr") else "") + \
            ("_im" if c.get("init_mem") else "") + ("_og" if c.get("out_gate") else "") + \
            ("_sgn" if c.get("signed") else "") + ("_kc" if c.get("keycheck") else "") + \
            (f"_flip{c['bitflip']}" if c.get("bitflip") else "") + \
            (f"_top{c['topk']}" if c.get("topk") else "") + \
            (f"_{c['corpus']}" if c.get("corpus", "shakespeare") != "shakespeare" else "") + \
            (f"_st{c['steps']}" if c.get("script") == "train_lm.py" and c.get("steps", 3000) != 3000 else "") + \
            (f"_an{c['anneal_frac']}" if c.get("anneal_frac") else "") + \
            (f"_lr{c['lr']:g}" if c.get("lr", 1e-3) != 1e-3 else "") + \
            (f"_d{c['d']}L{c['layers']}T{c['T']}" if c.get("d") else "") + \
            (f"_upd{c['updates']}" if c.get("updates") else "") + \
            (f"_{c['write']}" if c.get("write", "add") != "add" else "") + \
            (f"_rp{c['read_power']}" if c.get("read_power", 1) != 1 else "")
    if c.get("script") == "train_lm.py":
        name = f"lm_{c['mixer']}_F{c['feat']}" + (f"_nap{nap_of(c)}" if c["mixer"].startswith(("lut", "hybrid")) else "")
        return name + extra + f"_s{c.get('seed', 0)}"
    name = f"{c['mixer']}_n{c['n_pairs']}_F{c['feat']}"
    if c["mixer"].startswith(("lut", "hybrid")):
        name += f"_nap{nap_of(c)}"
    name += extra
    if c.get("steps", 3000) != 3000:
        name += f"_st{c['steps']}"
    if c.get("gap"):
        name += f"_gap{c['gap']}"
    if c.get("shift_k", 1) != 1:
        name += f"_k{c['shift_k']}"
    if c.get("anchor_per_seed"):
        name += "_ap"
    return name + f"_s{c.get('seed', 0)}"


def write_reports(grid, current=None):
    rows, seen = [], set()
    for c in grid:
        if run_name(c) in seen:
            continue
        seen.add(run_name(c))
        s = os.path.join(RUNS, run_name(c), "summary.json")
        if os.path.exists(s):
            r = json.load(open(s))
            rows.append(dict(block=c.get("block", ""), mixer=c["mixer"], n_pairs=c.get("n_pairs", "LM"), feat=c["feat"],
                             nap=nap_of(c) if c["mixer"].startswith("lut") else "", seed=c.get("seed", 0),
                             steps=c.get("steps", 3000), best_acc=r["best_acc"], final_acc=r["final_acc"],
                             train_s=r["train_s"], pause_s=r["thermal_pause_s"], max_cpu_temp=r["max_cpu_temp"],
                             name=r["name"]))
    if rows:
        with open(os.path.join(RUNS, f"results_{GRID}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    done = {r["name"] for r in rows}
    lines = [f"# Progress — {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"{len(done)}/{len(grid)} runs done." + (f" Running: `{current}`" if current else ""), "",
             "| block | model | pairs | memory | seed | steps | final acc | best acc | train min | max °C |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['block']} | {r['mixer']} | {r['n_pairs']} | {r['feat']} | {r['seed']} | {r['steps']} | "
                     f"{r['final_acc']:.4f} | {r['best_acc']:.4f} | {r['train_s'] / 60:.1f} | {r['max_cpu_temp']} |")
    pending = [run_name(c) for c in grid if run_name(c) not in done]
    if pending:
        lines += ["", "Pending: " + ", ".join(f"`{p}`" for p in pending)]
    text = "\n".join([f"Grid: {GRID}"] + lines) + "\n"
    open(os.path.join(RUNS, f"PROGRESS_{GRID}.md"), "w").write(text)
    open(os.path.join(RUNS, "PROGRESS.md"), "w").write(text)  # always the currently running grid
    if True:
        subprocess.run([PY, os.path.join(HERE, "build_report.py")], capture_output=True)  # refresh live HTML report


GRID = sys.argv[1] if len(sys.argv) > 1 else "night"


def main():
    grid = GRIDS[GRID]
    os.makedirs(RUNS, exist_ok=True)
    th = Thermal(log_path=os.path.join(RUNS, "thermal_between_runs.jsonl"))
    for c in grid:
        name = run_name(c)
        if os.path.exists(os.path.join(RUNS, name, "summary.json")):
            print(f"skip {name} (done)", flush=True)
            continue
        # agreed rule (2026-10-19): if seed 0 of this exact configuration ended below 10% recall, skip later
        # seeds -- except 6-bit drawer tables, whose failures are known to depend on the anchor draw
        if c.get("seed", 0) > 0 and c.get("script", "train.py") == "train.py" and not (c["mixer"].startswith("lut") and nap_of(c) >= 6):
            s0 = os.path.join(RUNS, run_name(dict(c, seed=0)), "summary.json")
            if os.path.exists(s0) and json.load(open(s0))["final_acc"] < 0.10:
                os.makedirs(os.path.join(RUNS, name), exist_ok=True)
                json.dump(dict(c, skipped="seed 0 ended below 10%"), open(os.path.join(RUNS, name, "SKIPPED.json"), "w"))
                print(f"skip {name} (seed 0 failed)", flush=True)
                continue
        write_reports(grid, current=f"{name} (cooling)")
        th.wait_power(why=f"before {name}")
        th.wait_below(COOL_TO, why=f"before {name}")
        write_reports(grid, current=name)
        print(f"=== {name}  start cpu {th.cpu_temp():.1f}C  {time.strftime('%H:%M:%S')}", flush=True)
        args = [PY, os.path.join(HERE, c.get("script", "train.py")), "--name", name, "--nap", str(nap_of(c))]
        for k, v in c.items():
            if k in ("block", "nap", "script"):
                continue
            if isinstance(v, bool):
                args += [f"--{k}"] if v else []
            else:
                args += [f"--{k}", str(v)]
        os.makedirs(os.path.join(RUNS, name), exist_ok=True)
        with open(os.path.join(RUNS, name, "train.log"), "w") as log:
            rc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"=== {name}  end rc={rc} cpu {th.cpu_temp():.1f}C  {time.strftime('%H:%M:%S')}", flush=True)
        write_reports(grid)
        if os.environ.get("LAB_PUSH"):  # two-Mac setup: publish each finished run to the shared repo
            g = lambda *a: subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True)
            g("add", os.path.join(RUNS, name), os.path.join(RUNS, f"results_{GRID}.csv"), os.path.join(RUNS, f"PROGRESS_{GRID}.md"))
            g("commit", "-m", f"run {name} ({os.uname().nodename.split('.')[0]})")
            for _ in range(4):
                if g("pull", "--rebase", "-q").returncode == 0 and g("push", "-q").returncode == 0:
                    break
                time.sleep(20)
    th.close()
    write_reports(grid)
    print(open(os.path.join(RUNS, f"PROGRESS_{GRID}.md")).read(), flush=True)


if __name__ == "__main__":
    main()
