"""Coordinate descent on the 153K teacher dataset, then TRUE-SNN validation.

Split is by SEED (0,1,2 train / 3 held-out), so held-out is whole independent rollouts, not
a slice of training trajectories.

Cell selection uses `obs_bucket` directly: the dequantisation table is strictly increasing,
so `dequant[b_a] > dequant[b_b]  <=>  b_a > b_b`. Comparing bucket indices is therefore
EXACTLY the encoder's comparison, ties included, with no float involved.

⚠️ These are teacher-visited (open-loop) states. High held-out agreement here says nothing
about closed-loop return -- that lesson is already paid for.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/walker2d-lut/src")
import tiny_lut_quantised_pipeline as QP                    # noqa: E402

NPZ = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/"
       "exp19_lut-lse-expmlpcrit-t32/deploy/quantised/"
       "walker2d_fastlut_lse_exp19_quantised.npz")
ACT = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/walker2d-spiking/"
       "deploy_quantised/spiking_lut_quantised_actor.npz")
DATA = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/walker2d-spiking/"
        "analysis/software_teacher_io_dataset_100k.npz")
# The fit's outputs; §4 of the write-up: these two are a PAIR and must never be used apart.
WOUT = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/walker2d-spiking/"
        "deploy_quantised/stage3_weights_bigdata.npy")
OOUT = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/walker2d-spiking/"
        "deploy_quantised/stage3_offset_bigdata.npy")
PHASE, BASE = 0.750, 13.0


class NpzView:
    def __init__(self, b, W):
        self._b, self._W, self.files = b, W, list(b.files)

    def __getitem__(self, k):
        return self._W if k == "weights" else self._b[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=600.0)
    ap.add_argument("--snn-states", type=int, default=6144)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Z, Q, D = np.load(NPZ), np.load(ACT), np.load(DATA)
    meta = json.loads(str(D["meta"]))
    per_seed = meta["envs"] * meta["steps"]
    tau = float(Z["tau_actor"])
    CL, LV = float(Z["out_quant_clip"]), int(Z["out_quant_levels"])
    ST = 2 * CL / (LV - 1)
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    pw = 1 << np.arange(A_.shape[1] - 1, -1, -1)
    W0 = Z["weights"].astype(np.float64)
    beta = Q["beta"].astype(np.float64)
    aff = Q["affine"].astype(np.float64)
    tau_eff = 1.0 / np.log((1.0 + 0.5 / float(Q["tau_m_out"])) ** 2)

    B = D["obs_bucket"].astype(np.int64)
    q_ref = D["action"].astype(np.float64)
    n_tr = 3 * per_seed
    idx_all = (((B[:, A_] - B[:, B_]) > 0).astype(np.int64) * pw).sum(-1)   # exact encoder
    idx_tr, idx_ho = idx_all[:n_tr], idx_all[n_tr:]
    q_tr, q_ho = q_ref[:n_tr], q_ref[n_tr:]
    print(f"dataset {len(B):,} pairs -> train {len(idx_tr):,} (seeds 0-2) / "
          f"held-out {len(idx_ho):,} (seed 3)")

    L0 = W0 / tau
    lo, hi = L0.min(), L0.max()
    step = (hi - lo) / 255.0
    L = lo + np.round((L0 - lo) / step) * step
    print(f"8-bit log-domain grid: step {step:.6f}")

    def decode(Lw, idx, off):
        S = np.exp(Lw[np.arange(32)[None, :], idx]).sum(1)
        T = np.ceil(-tau_eff * np.log(beta[None, :] * S) + PHASE) + BASE
        mu = aff[:, 0][None, :] * T + off[None, :]
        return np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)

    def stats(q, ref):
        lev = np.rint((q - ref) / ST).astype(int)
        return dict(exact=[float((lev[:, o] == 0).mean()) for o in range(6)],
                    within1=[float((np.abs(lev[:, o]) <= 1).mean()) for o in range(6)],
                    mean_signed=[float(lev[:, o].mean()) for o in range(6)],
                    hist=[[float((lev[:, o] == k).mean()) for k in (-2, -1, 0, 1, 2)]
                          for o in range(6)])

    S_tr0 = np.exp(L[np.arange(32)[None, :], idx_tr]).sum(1)
    T0 = np.ceil(-tau_eff * np.log(beta[None, :] * S_tr0) + PHASE) + BASE
    mu_sw_tr = 32 * tau * np.log(np.exp(W0[np.arange(32)[None, :], idx_tr] / tau).mean(1))
    off = np.median(mu_sw_tr - aff[:, 0][None, :] * T0, axis=0)
    base_ho = stats(decode(L, idx_ho, off), q_ho)
    print(f"\nGATE (scorer baseline, held-out): exact "
          f"{[round(v*100,2) for v in base_ho['exact']]}")
    print(f"  mean signed {[round(v,4) for v in base_ho['mean_signed']]}")
    ok = min(base_ho["exact"]) > 0.70 and max(base_ho["mean_signed"]) < 0.0
    print(f"  reproduces true-SNN behaviour (exact>70%, all-negative bias): "
          f"{'PASS' if ok else 'FAIL'}")
    assert stats(decode(L, idx_ho, off), q_ho)["exact"] == base_ho["exact"]
    print("  zero-step = baseline: PASS")
    if not ok:
        json.dump(dict(gate_passed=False, baseline=base_ho), open(a.out, "w"), indent=1)
        return

    # ---- coordinate descent -----------------------------------------------------------
    def loss(S, o, ref):
        T = np.ceil(-tau_eff * np.log(beta[o] * S) + PHASE) + BASE
        mu = aff[o, 0] * T + off[o]
        q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        return np.abs(np.rint((q - ref) / ST)).sum()

    S = np.exp(L[np.arange(32)[None, :], idx_tr]).sum(1)
    Lg = np.round((L - lo) / step)
    Lc = L.copy()
    traj = [float(sum(loss(S[:, o], o, q_tr[:, o]) for o in range(6)))]
    print(f"\nsweep 0 loss {traj[0]:,.0f}")
    t0, moved, stop = time.time(), 0, False
    for sw in range(a.sweeps):
        for o in range(6):
            for t in range(32):
                col = idx_tr[:, t]
                order = np.argsort(col, kind="stable")
                bnd = np.searchsorted(col[order], np.arange(65))
                for k in range(64):
                    sl = order[bnd[k]:bnd[k + 1]]
                    if sl.size == 0:
                        continue
                    Sm, rm = S[sl, o], q_tr[sl, o]
                    cur = loss(Sm, o, rm)
                    e_old = np.exp(Lc[t, k, o])
                    best = (cur, 0, None)
                    for d in list(range(-a.kmax, 0)) + list(range(1, a.kmax + 1)):
                        Sn = Sm - e_old + np.exp(lo + (Lg[t, k, o] + d) * step)
                        if (Sn <= 0).any():
                            continue
                        c = loss(Sn, o, rm)
                        if c < best[0]:
                            best = (c, d, Sn)
                    if best[1]:
                        Lg[t, k, o] += best[1]
                        Lc[t, k, o] = lo + Lg[t, k, o] * step
                        S[sl, o] = best[2]
                        moved += 1
                if time.time() - t0 > a.budget_s:
                    stop = True; break
            if stop:
                break
        traj.append(float(sum(loss(S[:, o], o, q_tr[:, o]) for o in range(6))))
        print(f"sweep {sw+1} loss {traj[-1]:,.0f}  moved {moved:,}  {time.time()-t0:.0f}s"
              + ("  [budget]" if stop else ""))
        if stop:
            break

    S2 = np.exp(Lc[np.arange(32)[None, :], idx_tr]).sum(1)
    T2 = np.ceil(-tau_eff * np.log(beta[None, :] * S2) + PHASE) + BASE
    off2 = np.median(mu_sw_tr - aff[:, 0][None, :] * T2, axis=0)
    aft_ho = stats(decode(Lc, idx_ho, off2), q_ho)
    drift = float(np.abs(Lc - L).max())
    print(f"\n=== HELD-OUT (scorer) ===")
    for o in range(6):
        print(f"  dim {o}: exact {base_ho['exact'][o]*100:6.2f}% -> "
              f"{aft_ho['exact'][o]*100:6.2f}%   signed "
              f"{base_ho['mean_signed'][o]:+7.4f} -> {aft_ho['mean_signed'][o]:+7.4f}   "
              f"hist {[round(v*100,1) for v in aft_ho['hist'][o]]}")
    print(f"  overall {np.mean(base_ho['exact'])*100:.2f}% -> "
          f"{np.mean(aft_ho['exact'])*100:.2f}%   within1 "
          f"{np.mean(aft_ho['within1'])*100:.3f}%   moved {moved:,}/12,288  drift {drift:.4f}")

    # ---- TRUE SNN validation on a held-out slice, GPU ---------------------------------
    ns = min(a.snn_states, len(idx_ho))
    Xho = D["obs_norm"][n_tr:n_tr + ns].astype(np.float64)
    edges = Z["in_quant_edges"]
    snn = {}
    for name, Wu in (("baseline", W0), ("optimised", Lc * tau)):
        net, ids, _, ntk, nn_, _, _, _, _ = QP.build(
            NpzView(Z, Wu), list(range(6)), False, float(Q["tau_m_out"]), "cuda", 6, 3.0, True)
        T, TL = [], []
        for s0 in range(0, ns, 64):
            tk = QP.encode_gauss(Xho[s0:s0 + 64], edges)
            o_ = QP.run(net, ids, tk, ntk, "cuda")
            T.append(o_[6].astype(np.float64)); TL.append(tk.max(1).astype(np.float64))
        T, TL = np.concatenate(T), np.concatenate(TL)
        offs = off if name == "baseline" else off2
        mu = aff[:, 0][None, :] * (T - TL[:, None]) + offs[None, :]
        q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        snn[name] = stats(q, q_ho[:ns])
        print(f"\nTRUE SNN {name}: exact {[round(v*100,2) for v in snn[name]['exact']]}  "
              f"overall {np.mean(snn[name]['exact'])*100:.2f}%")
        print(f"  signed {[round(v,4) for v in snn[name]['mean_signed']]}")

    # deploy_quantised/ is an output dir, not tracked — create it on first run.
    os.makedirs(os.path.dirname(WOUT), exist_ok=True)
    np.save(WOUT, Lc * tau)
    np.save(OOUT, off2)
    print(f"offset shift vs baseline (level units): "
          f"{[round(float((off2[o]-off[o])/ST),4) for o in range(6)]}")
    print("saved stage3_weights_bigdata.npy")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(gate_passed=True, n_train=len(idx_tr), n_heldout=len(idx_ho),
                   scorer_before=base_ho, scorer_after=aft_ho, moved=int(moved),
                   drift=drift, loss_traj=traj, true_snn=snn, snn_states=int(ns),
                   caveat="teacher-visited (open-loop) states; says nothing about return"),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
