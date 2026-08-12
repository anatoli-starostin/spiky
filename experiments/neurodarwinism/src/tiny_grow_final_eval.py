"""exp012 growable nets: score and dissect the run's properly-selected model.

The headline is the FINAL EWMA LEADER -- the member selection settles on, chosen on training
batches alone. Reports the pure held-out MSE (not the fitness), the fitness broken into its
three terms, the network's size and fan-out profile, and what the surviving neurons do.
"""
import argparse
import json

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load
from harness import LatencyEncoder
from tiny_grow_evolve import load_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lam", type=float, default=0.35)
    ap.add_argument("--mu", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pool, ewma, age, rnd, hist, best, _ = load_ckpt(a.ckpt)
    fin = np.where(np.isfinite(ewma))[0]
    lead = int(fin[np.argmin(ewma[fin])])
    g = pool[lead]

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    tgt = T.target_offsets(Yv)

    s = G.score(G.build([g], device=a.device), Xv, Yv, enc, genomes=[g])
    first = s["first"][:, 0, :]
    calib = s["calibrated"][:, 0, :]
    mse = float(s["mse"][0])
    r, ceil = T.affine_ceiling_and_r(calib, tgt)
    aff_a, aff_b = G.affine_of(g)

    # TWO ceilings, because they are not the same number and only one of them is reachable.
    # `ceil` above is fitted ON the held-out set and scored on it -- an optimistic bound, and
    # the one this chapter has been quoting. Refitting the same 12 parameters on a TRAINING
    # batch and scoring on held-out gives what anything selected on training data can actually
    # reach; the pre-flight measured that gap at ~4.8 MSE (train/held-out distribution shift,
    # not overfitting -- the held-out split is the trajectory's tail).
    from data import sample_batch
    Xb2, Yb2, _ = sample_batch(Xp, Yp, 2000, a.seed, 7)
    raw_tr = G.score(G.build([g], device=a.device), Xb2, Yb2, enc)["first"][:, 0, :]
    at, bt = G.analytic_affine(raw_tr, T.target_offsets(Yb2))
    ceil_honest = float((((at * first + bt) - tgt) ** 2).mean())
    per = []
    for d in range(first.shape[1]):
        p, t = calib[:, d], tgt[:, d]
        rr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else 0.0
        per.append(dict(dim=d, r=rr, bias2=float((p.mean() - t.mean()) ** 2),
                        scale_err=float((p.std() - t.std()) ** 2)))
    b2 = float(np.mean([x["bias2"] for x in per]))
    sc = float(np.mean([x["scale_err"] for x in per]))

    # ---- size and fan-out
    act_e = np.where(g["act_exc"])[0]
    act_i = np.where(g["act_inh"])[0]
    fo = g["mask"].sum(1)
    fi = g["mask"].sum(0)
    terms = G.cost_terms(g, mse, a.lam, a.mu)
    rows = []
    for s_ in act_e:
        rows.append(dict(unit=f"E{s_}", slot=int(s_), kind="exc",
                         fanout=int(fo[G.N_IN + s_]), fanin=int(fi[s_]),
                         is_seed_slot=bool(s_ < 8)))
    for s_ in act_i:
        rows.append(dict(unit=f"I{s_}", slot=int(s_), kind="inh",
                         fanout=int(fo[G.N_IN + G.N_EXC_MAX + s_]),
                         fanin=int(fi[G.N_EXC_MAX + s_]), is_seed_slot=bool(s_ < 2)))

    # ---- live-path check: is each hidden unit downstream of an input AND upstream of an out?
    m = g["mask"]
    reach = np.zeros(G.N_SRC, bool)
    reach[G.R_IN] = True
    for _ in range(G.N_SRC):
        new = reach.copy()
        for i in np.where(reach)[0]:
            for j in np.where(m[i])[0]:
                if j < G.N_EXC_MAX:
                    new[G.N_IN + j] = True
                elif j < G.N_EXC_MAX + G.N_INH_MAX:
                    new[G.N_IN + G.N_EXC_MAX + (j - G.N_EXC_MAX)] = True
        if (new == reach).all():
            break
        reach = new
    to_out = np.zeros(G.N_SRC, bool)
    for i in range(G.N_SRC):
        if m[i, G.C_OUT].any():
            to_out[i] = True
    for _ in range(G.N_SRC):
        new = to_out.copy()
        for i in range(G.N_SRC):
            for j in np.where(m[i])[0]:
                if j < G.N_EXC_MAX and to_out[G.N_IN + j]:
                    new[i] = True
                elif (G.N_EXC_MAX <= j < G.N_EXC_MAX + G.N_INH_MAX
                      and to_out[G.N_IN + G.N_EXC_MAX + (j - G.N_EXC_MAX)]):
                    new[i] = True
        if (new == to_out).all():
            break
        to_out = new
    for x in rows:
        i = (G.N_IN + x["slot"]) if x["kind"] == "exc" else (G.N_IN + G.N_EXC_MAX + x["slot"])
        x["live"] = bool(reach[i] and to_out[i])

    # ---- capacity utilisation: wired is not the same as used, and used is not the same as
    # firing. A slot can be active, have edges in and out, and still never spike.
    import torch
    from spiky.spnet.spnet import NeuronDataType
    Hf = G.build([g], device=a.device)
    sp, ids = Hf["spnet"], Hf["ids"]
    Bf = min(1024, Xv.shape[0])
    cols = ids[2]
    tk = enc(Xv[:Bf])
    va = np.zeros((Bf, T.T_IN, cols.size), np.float32)
    for b in range(Bf):
        for j in range(T.N_IN):
            va[b, tk[b, j], j::T.N_IN] = 200.0
    sp.process_ticks(n_ticks_to_process=T.N_TICKS, batch_size=Bf, n_input_ticks=T.T_IN,
                     input_values=torch.as_tensor(va, device=a.device),
                     sparse_input=T._sparse_ids(cols, Bf, T.T_IN, a.device),
                     do_train=False, do_record_voltage=False, do_reset_context=True,
                     _stdp_period=32)
    util = {}
    for k, nm, n_slot in ((0, "exc", G.N_EXC_MAX), (1, "inh", G.N_INH_MAX)):
        oid = torch.as_tensor(np.ascontiguousarray(ids[k], dtype=np.int32), device=a.device)
        Rf = sp.export_neuron_data(oid, Bf, NeuronDataType.Spike, 0, T.N_TICKS - 1)
        per = Rf.ne(0).sum(-1).float().mean(0).cpu().numpy()      # spikes/episode per slot
        act = g["act_exc"] if nm == "exc" else g["act_inh"]
        off = G.N_IN if nm == "exc" else G.N_IN + G.N_EXC_MAX
        cof = 0 if nm == "exc" else G.N_EXC_MAX
        fanout = np.array([int(m[off + s].sum()) for s in range(n_slot)])
        fanin = np.array([int(m[:, cof + s].sum()) for s in range(n_slot)])
        wired = (fanin > 0) & (fanout > 0) & act
        fires = (per > 0) & act
        util[nm] = dict(n_slots=n_slot, n_active=int(act.sum()),
                        n_wired_both_ways=int(wired.sum()), n_firing=int(fires.sum()),
                        n_wired_but_silent=int((wired & ~fires).sum()),
                        mean_spikes_per_episode_active=float(per[act].mean()),
                        mean_spikes_per_episode_wired=float(per[wired].mean())
                        if wired.any() else 0.0,
                        spikes_per_episode=[round(float(x), 3) for x in per[:n_slot]])
    out_util = util

    out = dict(ckpt=a.ckpt, round=rnd, leader_index=lead, leader_ewma=float(ewma[lead]),
               capacity_utilisation=out_util,
               heldout_mse=mse, constant_baseline=T.constant_baseline(Yv),
               fitness_terms=terms, tau=float(s["tau"][0]), mean_abs_r=r,
               affine_ceiling=ceil, affine_ceiling_honest_trainfit=ceil_honest,
               analytic_a_trainfit=at.tolist(), analytic_b_trainfit=bt.tolist(),
               silent=float(s["silent"][0]),
               n_distinct=int(s["n_distinct"][0]), mse_action=float(s["mse_action"][0]),
               bias2=b2, scale_err=sc, resid=mse - b2 - sc, per_dim=per,
               affine_a=aff_a.tolist(), affine_b=aff_b.tolist(),
               size=G.genome_stats(g), units=rows,
               fanout_distribution=sorted(int(x) for x in fo[fo > 0]),
               n_over_cap=int((fo > terms["cap"]).sum()),
               batch_champion_heldout=best.get("heldout_mse"))
    print(json.dumps(T.jsonable({k: v for k, v in out.items()
                                 if k not in ("per_dim", "units")}), indent=1))
    print("\nUNITS")
    for x in rows:
        print("  ", x)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(out), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
