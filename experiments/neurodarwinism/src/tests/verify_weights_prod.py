"""STEP 0: is our REAL production weights placement correct at 40 metas, engine gs=2?

steady_state.build_pool goes through synapse_growth._grow_explicit(weights=), whose aligner
follows the group chain as of PR #94. This checks it on a realistic multi-meta explicit
genome, keyed on (src,tgt), with EVERY EXCITATORY EDGE GIVEN A UNIQUE WEIGHT so a
misplacement cannot hide behind a repeated value. (It predates that fix, when build_pool
used a chain-following copy that lived in harness.py; the check is unchanged, the path
under it is now the library's.)

Inhibitory edges keep RES_W_INH: their metas pin min==max==-5, so unique values there would
be clamped and register as false mismatches. They are checked separately for being exactly -5.
"""
# tests live one level below src/; make the sibling modules importable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse

import numpy as np
import torch

import steady_state as S

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--metas", type=int, default=40, help="40 = production (20 exc + 20 inh)")
ap.add_argument("--gs", type=int, default=2)
ap.add_argument("--w-max", type=float, default=30.0)
a = ap.parse_args()

n_delays = max(1, a.metas // 2) if a.metas > 1 else 1
S.GROUP_SIZE = a.gs
_orig = S.stage2_metas
S.stage2_metas = lambda lr, wm, group_size=a.gs, backward_group_size=32: _orig(
    lr, wm, group_size=group_size, backward_group_size=backward_group_size)

genomes = [S.seed_genome(np.random.default_rng(i), a.w_max) for i in range(a.k)]
for g in genomes:
    # collapse delays into n_delays distinct buckets -> that many excitatory metas in use
    g["delay"] = S.D_MIN + (g["delay"] - S.D_MIN) % n_delays
    exc = g["src_pool"] != S.INH
    # unique weight per excitatory edge, inside the meta bounds [0, 1.5*w_max]
    g["weight"] = g["weight"].astype(np.float64)
    g["weight"][exc] = np.linspace(0.001, 1.4 * a.w_max, int(exc.sum()))
    g["weight"][~exc] = S.RES_W_INH

used = sorted({int(d) for g in genomes for d in np.unique(g["delay"])})
print(f"K={a.k} engine_gs={a.gs} delay buckets in use={len(used)} "
      f"(-> {len(used)} exc metas + {len(used)} inh metas)")

h = S.build_pool(genomes, "cuda", seed=1, stdp_lr=0.01, w_max=a.w_max)
torch.cuda.synchronize()
print(f"BUILD-OK {h['n_syn']:,} synapses")

ids = h["ids"]
all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device="cuda")
n = h["spnet"].count_synapses(all_ids, True)
b = [torch.zeros(n, dtype=t, device="cuda") for t in
     (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
h["spnet"].export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
es, em, ew, ed, et = (x.cpu().numpy() for x in b)
got = {(int(s), int(t)): float(w) for s, t, w in zip(es, et, ew)}

base = {S.EXC: S.N_EXC, S.INH: S.N_INH, S.INP: S.N_IN, S.OUTP: S.N_OUT}
exact = wrong = missing = 0
inh_ok = inh_bad = 0
worst = []
for c, g in enumerate(genomes):
    gs_ = np.empty(g["weight"].size, np.int64)
    gt_ = np.empty_like(gs_)
    for p in (S.EXC, S.INH, S.INP):
        m = g["src_pool"] == p
        if m.any():
            gs_[m] = ids[p][c * base[p] + g["src_idx"][m]]
    for p in (S.EXC, S.INH, S.OUTP):
        m = g["tgt_pool"] == p
        if m.any():
            gt_[m] = ids[p][c * base[p] + g["tgt_idx"][m]]
    for i in range(g["weight"].size):
        key = (int(gs_[i]), int(gt_[i]))
        want = float(g["weight"][i])
        if key not in got:
            missing += 1
            continue
        if g["src_pool"][i] == S.INH:
            if abs(got[key] - S.RES_W_INH) < 1e-6:
                inh_ok += 1
            else:
                inh_bad += 1
        elif abs(got[key] - want) <= 1e-3:
            exact += 1
        else:
            wrong += 1
            if len(worst) < 3:
                worst.append((key, want, got[key]))

tot = exact + wrong
print(f"EXCITATORY: exact {exact:,}/{tot:,} ({exact / max(tot,1):.2%})  "
      f"wrong {wrong:,}  missing {missing:,}")
print(f"INHIBITORY: pinned at -5: {inh_ok:,}   not pinned: {inh_bad:,}")
for k, w1, w2 in worst:
    print(f"    edge {k}: want {w1:.4f} got {w2:.4f}")
print("VERDICT:", "PLACEMENT CORRECT" if wrong == 0 and missing == 0 else "PLACEMENT CORRUPTED")
