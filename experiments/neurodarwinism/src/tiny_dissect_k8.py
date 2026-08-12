"""exp012: dissect the K=8 diagonal-LS leader -- structure, function, and what actually matters.

The decisive experiment here is the last one: freeze the wiring and the delays, redraw the
weights on the {-1,0,+1} grid many times, and see how much of the performance survives. If
MSE barely moves, the network is a wiring-and-delay machine and the weights are close to
incidental. If it collapses, the exact weight assignment is doing the work.

Everything is scored the way the run scored it: a diagonal least-squares readout fitted on a
TRAINING batch and carried unchanged to held-out.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

CK = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
      "exp012_tiny-direct-genome/run_diagls_k8/ck_P0.npz")


def setup():
    G.set_out_per_target(8, "mean")
    G.set_weight_levels([-1.0, 0.0, 1.0])
    G.set_delay_levels(list(range(1, 64, 2)))
    G.QUANTIZED = True
    G.FANOUT_CAP = 16
    G.MAX_EPISODE_BATCH = 256


def cls_of(i, j):
    src = "in" if i < G.N_IN else ("exc" if i < G.N_IN + G.N_EXC_MAX else "inh")
    tgt = ("exc" if j < G.N_EXC_MAX else
           "inh" if j < G.N_EXC_MAX + G.N_INH_MAX else "out")
    return f"{src}->{tgt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n-redraw", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setup()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    from tiny_grow_evolve import load_ckpt
    pool, ewma, *_ = load_ckpt(a.ckpt)
    fin = np.where(np.isfinite(ewma))[0]
    g = pool[int(fin[np.argmin(ewma[fin])])]

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, 7)
    tgt = T.target_offsets(Yv)
    chance = T.constant_baseline(Yv)

    import torch

    def evaluate(gg):
        """held-out MSE with the diagonal-LS readout fitted on TRAINING.

        This routine is called ~100 times (ablations, knockouts, redraws). Each call builds a
        fresh SpikingNet with its own device buffers, and without an explicit free they
        accumulate until the engine dies with cudaErrorIllegalAddress -- which is what
        happened on the first attempt, at roughly the tenth build.
        """
        if gg["mask"].sum() == 0:
            return float(chance), None          # a net with no synapses predicts nothing
        H = G.build([gg], device=a.device)
        st = G.score(H, Xb, Yb, enc, genomes=[gg], readout="diagls")
        sv = G.score(H, Xv, Yv, enc, genomes=[gg], readout="diagls",
                     readout_map=st["readout_map"])
        out = float(sv["mse"][0])
        del H, st
        torch.cuda.empty_cache()
        return out, sv

    R = dict(chance=chance)
    base, sv = evaluate(g)
    R["baseline_mse"] = base

    # ---------------------------------------------------------------- STRUCTURE
    m = g["mask"]
    r, c = np.nonzero(m)
    w = g["weight"][r, c]
    d = g["delay"][r, c]
    kinds = np.array([cls_of(i, j) for i, j in zip(r, c)])
    by_class = {}
    for k in sorted(set(kinds)):
        sel = kinds == k
        by_class[k] = dict(n=int(sel.sum()),
                           n_pos=int((w[sel] > 0).sum()), n_neg=int((w[sel] < 0).sum()),
                           n_zero=int((w[sel] == 0).sum()),
                           delay_mean=float(d[sel].mean()), delay_median=float(np.median(d[sel])),
                           delay_min=int(d[sel].min()), delay_max=int(d[sel].max()),
                           frac_delay_ge_32=float((d[sel] >= 32).mean()))
    fo = m.sum(1)
    fi = m.sum(0)
    hid_fo = G.hidden_fanout(g)
    R["structure"] = dict(
        n_synapses=int(m.sum()), by_class=by_class,
        weight_levels=dict(Counter(np.round(w, 3).tolist())),
        frac_negative=float((w < 0).mean()), frac_zero=float((w == 0).mean()),
        fanout_hidden=dict(max=int(hid_fo.max()), at_cap=int((hid_fo == 16).sum()),
                           mean=float(hid_fo.mean()), n_hidden=len(hid_fo)),
        fanout_inputs=dict(max=int(fo[:G.N_IN].max()), mean=float(fo[:G.N_IN].mean())),
        fanin_outputs=dict(min=int(fi[G.C_OUT].min()), max=int(fi[G.C_OUT].max()),
                           mean=float(fi[G.C_OUT].mean()),
                           n_with_zero_fanin=int((fi[G.C_OUT] == 0).sum())),
        delay_overall=dict(mean=float(d.mean()), median=float(np.median(d)),
                           frac_ge_32=float((d >= 32).mean()),
                           hist_bins_of_8=[int(((d >= lo) & (d < lo + 8)).sum())
                                           for lo in range(1, 65, 8)]))

    # reachability: which hidden units are downstream of an input AND upstream of an output
    fwd = {i: [] for i in range(G.N_SRC)}
    for i, j in zip(r, c):
        if j < G.N_EXC_MAX:
            fwd[i].append(G.N_IN + j)
        elif j < G.N_EXC_MAX + G.N_INH_MAX:
            fwd[i].append(G.N_IN + G.N_EXC_MAX + (j - G.N_EXC_MAX))
    seen = set(range(G.N_IN))
    frontier = list(range(G.N_IN))
    while frontier:
        n = frontier.pop()
        for x in fwd[n]:
            if x not in seen:
                seen.add(x)
                frontier.append(x)
    to_out = {i for i in range(G.N_SRC) if m[i, G.C_OUT].any()}
    for _ in range(G.N_SRC):
        add = {i for i in range(G.N_SRC) for x in fwd[i] if x in to_out}
        if add <= to_out:
            break
        to_out |= add
    live = [i for i in range(G.N_IN, G.N_SRC) if i in seen and i in to_out]
    R["structure"]["hidden_live"] = len(live)
    R["structure"]["hidden_total"] = G.N_EXC_MAX + G.N_INH_MAX

    # ---------------------------------------------------------------- OUTPUT GROUPS
    raw = sv["raw_neurons"][:, 0, :]                      # [B, 48]
    grp = raw.reshape(-1, G.N_TARGET, G.OUT_PER_TARGET)
    R["output_groups"] = []
    for t in range(G.N_TARGET):
        cols = [G.N_EXC_MAX + G.N_INH_MAX + t * G.OUT_PER_TARGET + k
                for k in range(G.OUT_PER_TARGET)]
        R["output_groups"].append(dict(
            target=t, fanin_per_member=[int(fi[cc]) for cc in cols],
            n_members_wired=int(sum(fi[cc] > 0 for cc in cols)),
            n_members_varying=int(sum(grp[:, t, k].std() > 1e-9
                                      for k in range(G.OUT_PER_TARGET))),
            member_sd=[round(float(grp[:, t, k].std()), 3)
                       for k in range(G.OUT_PER_TARGET)],
            agg_sd=float(grp[:, t, :].mean(-1).std())))

    # ---------------------------------------------------------------- FUNCTION
    cal = sv["calibrated"][:, 0, :]
    per = []
    for t in range(G.N_TARGET):
        p_, q_ = cal[:, t], tgt[:, t]
        per.append(dict(target=t, mse=float(((p_ - q_) ** 2).mean()),
                        bias2=float((p_.mean() - q_.mean()) ** 2),
                        scale_err=float((p_.std() - q_.std()) ** 2),
                        r=float(np.corrcoef(p_, q_)[0, 1]) if p_.std() > 1e-9 else 0.0,
                        target_sd=float(q_.std()), pred_sd=float(p_.std())))
    R["per_target"] = per

    # hidden first-spike vs each target
    from tiny_ceiling import features
    fv = features(g, Xv, enc, a.device)
    hf = np.column_stack([fv["exc_first"], fv["inh_first"]])
    C = np.zeros((hf.shape[1], G.N_TARGET))
    for i in range(hf.shape[1]):
        if hf[:, i].std() > 1e-9:
            for t in range(G.N_TARGET):
                C[i, t] = np.corrcoef(hf[:, i], tgt[:, t])[0, 1]
    R["hidden_target_corr"] = dict(
        max_abs=float(np.abs(C).max()),
        mean_abs=float(np.abs(C).mean()),
        n_specialised=int(((np.abs(C).max(1) > 0.3)
                           & (np.abs(C).max(1) > 2 * np.sort(np.abs(C), 1)[:, -2])).sum()),
        n_with_any_corr_gt_03=int((np.abs(C).max(1) > 0.3).sum()),
        per_neuron_max=[round(float(x), 3) for x in np.abs(C).max(1)])

    # ---------------------------------------------------------------- ABLATIONS
    abl = {}
    for frac in (0.1, 0.25, 0.5, 0.75):
        h = {k: v.copy() if hasattr(v, "copy") else v for k, v in g.items()}
        mag = np.abs(h["weight"])
        thr = np.quantile(mag[h["mask"]], frac)
        h["mask"] = h["mask"] & ~(mag <= thr)
        abl[f"prune_weakest_{int(frac * 100)}pct"] = dict(
            n_syn=int(h["mask"].sum()), mse=evaluate(h)[0])
    for nm, sel in (("no_inhibition", (G.R_INH, slice(None))),
                    ("no_recurrence_exc", (slice(G.N_IN, None), G.C_EXC)),
                    ("no_hidden_to_hidden", (slice(G.N_IN, None), slice(0, G.N_EXC_MAX + G.N_INH_MAX)))):
        h = {k: v.copy() if hasattr(v, "copy") else v for k, v in g.items()}
        h["mask"][sel] = False
        abl[nm] = dict(n_syn=int(h["mask"].sum()), mse=evaluate(h)[0])
    R["ablations"] = abl

    # per-neuron knockout
    ko = []
    for s in range(G.N_EXC_MAX + G.N_INH_MAX):
        h = {k: v.copy() if hasattr(v, "copy") else v for k, v in g.items()}
        if s < G.N_EXC_MAX:
            h["mask"][G.N_IN + s, :] = False
            h["mask"][:, s] = False
            lab = f"E{s}"
        else:
            k2 = s - G.N_EXC_MAX
            h["mask"][G.N_IN + G.N_EXC_MAX + k2, :] = False
            h["mask"][:, G.N_EXC_MAX + k2] = False
            lab = f"I{k2}"
        ko.append(dict(unit=lab, mse=evaluate(h)[0]))
    for x in ko:
        x["delta"] = x["mse"] - base
    R["knockout"] = sorted(ko, key=lambda x: -x["delta"])

    # ---------------------------------------------------------------- THE KEY TEST
    # freeze wiring + delays, redraw weights on the grid. Dale-correct by construction.
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(a.n_redraw):
        h = {k: v.copy() if hasattr(v, "copy") else v for k, v in g.items()}
        npos = G.N_IN + G.N_EXC_MAX
        wn = np.empty((G.N_SRC, G.N_TGT))
        wn[:npos] = rng.choice(G.QUANT_POS, (npos, G.N_TGT))
        wn[npos:] = rng.choice(G.QUANT_NEG, (G.N_SRC - npos, G.N_TGT))
        h["weight"] = wn
        G.enforce(h)
        draws.append(evaluate(h)[0])
    draws = np.array(draws)
    # control: redraw the WIRING too (same synapse count), keeping delays
    ctrl = []
    for _ in range(min(a.n_redraw, 20)):
        h = {k: v.copy() if hasattr(v, "copy") else v for k, v in g.items()}
        legal = G.LEGAL & G.active_cells(h)
        flat = np.flatnonzero(legal)
        pick = rng.choice(flat, int(m.sum()), replace=False)
        nm2 = np.zeros(G.N_SRC * G.N_TGT, bool)
        nm2[pick] = True
        h["mask"] = nm2.reshape(G.N_SRC, G.N_TGT)
        G.enforce(h)
        ctrl.append(evaluate(h)[0])
    ctrl = np.array(ctrl)
    R["weights_vs_skeleton"] = dict(
        baseline=base, n_draws=int(len(draws)),
        weight_redraw=dict(mean=float(draws.mean()), sd=float(draws.std()),
                           min=float(draws.min()), max=float(draws.max()),
                           frac_below_chance=float((draws < chance).mean())),
        wiring_redraw=dict(mean=float(ctrl.mean()), sd=float(ctrl.std()),
                           min=float(ctrl.min()), max=float(ctrl.max()),
                           frac_below_chance=float((ctrl < chance).mean())),
        weight_redraw_all=[round(float(x), 3) for x in draws],
        wiring_redraw_all=[round(float(x), 3) for x in ctrl])

    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(json.dumps(T.jsonable({k: v for k, v in R.items()
                                 if k not in ("knockout", "hidden_target_corr",
                                              "output_groups", "structure")}), indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
