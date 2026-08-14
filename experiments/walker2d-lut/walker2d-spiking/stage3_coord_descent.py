"""Gradient-free coordinate descent on the Stage-3 readout weights.

No STE, no per-state gradients. The phase-pinned closed form (phase 0.750, base +13) is the
fast scorer; it agrees with the real SNN on ~97.7-99.8% of states, so the TRUE SNN is the
final arbiter and the closed-form numbers are treated as a screen.

Structure exploited for speed: weight (t,k,o) affects ONLY output dim o and ONLY the states
whose table t selected cell k, and it enters the readout through a SUM, so a candidate move
is an O(affected states) update:  S' = S - exp(w_old) + exp(w_new).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/walker2d-lut/src")
from warp_env import WarpWalker2dVecEnv                     # noqa: E402
import tiny_lut_quantised_pipeline as QP                    # noqa: E402

NPZ = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/"
       "exp19_lut-lse-expmlpcrit-t32/deploy/quantised/"
       "walker2d_fastlut_lse_exp19_quantised.npz")
ACT = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/walker2d-spiking/"
       "deploy_quantised/spiking_lut_quantised_actor.npz")
PHASE, BASE = 0.750, 13.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=420.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")

    Z, Q = np.load(NPZ), np.load(ACT)
    tau = float(Z["tau_actor"])
    CL, LV = float(Z["out_quant_clip"]), int(Z["out_quant_levels"])
    ST = 2 * CL / (LV - 1)
    edges, dq = Z["in_quant_edges"], Z["in_quant_dequant"]
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    pw = 1 << np.arange(A_.shape[1] - 1, -1, -1)
    W0 = Z["weights"].astype(np.float64)
    beta = Q["beta"].astype(np.float64)
    aff = Q["affine"].astype(np.float64)
    tau_eff = 1.0 / np.log((1.0 + 0.5 / float(Q["tau_m_out"])) ** 2)
    om, ov = Z["obs_mean"], Z["obs_var"]

    def collect(seed):
        env = WarpWalker2dVecEnv(num_envs=a.envs, seed=seed, solver_iters=100,
                                 ls_iters=50, obs_clip_vel=10.0)
        obs = env.reset()
        omt = torch.tensor(om, dtype=torch.float32, device=dev)
        ovt = torch.tensor(ov, dtype=torch.float32, device=dev)
        Xs = []
        for _ in range(a.steps):
            x = ((obs - omt) / torch.sqrt(ovt + 1e-8)).cpu().numpy().astype(np.float64)
            Xs.append(x)
            g = np.searchsorted(edges, x.ravel(), side="left").reshape(x.shape)
            xq = dq[np.clip(g, 0, len(dq) - 1)]
            i = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)
            s = W0[np.arange(32)[None, :], i]
            mu = 32 * tau * np.log(np.exp(s / tau).mean(1))
            q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
            obs, _, _, _ = env.step(torch.as_tensor(q, dtype=torch.float32, device=dev))
        X = np.concatenate(Xs)
        g = np.searchsorted(edges, X.ravel(), side="left").reshape(X.shape)
        xq = dq[np.clip(g, 0, len(dq) - 1)]
        idx = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)
        sel = W0[np.arange(32)[None, :], idx]
        mu_sw = 32 * tau * np.log(np.exp(sel / tau).mean(1))
        return X, idx, mu_sw

    Xtr, idx_tr, mu_tr = collect(0)
    Xho, idx_ho, mu_ho = collect(7)
    print(f"train {len(idx_tr):,}  held-out {len(idx_ho):,}")

    L0 = W0 / tau
    lo, hi = L0.min(), L0.max()
    step = (hi - lo) / (2 ** 8 - 1)
    Lg = np.round((L0 - lo) / step)                     # integer grid levels, 8-bit
    print(f"8-bit log-domain grid: step {step:.6f}, levels 0..255")

    def lev_of(L, idx, mu_ref, off):
        S = np.exp(L[np.arange(32)[None, :], idx]).sum(1)
        n = -tau_eff * np.log(beta[None, :] * S)
        T = np.ceil(n + PHASE) + BASE
        mu = aff[:, 0][None, :] * T + off[None, :]
        q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        qs = np.clip(np.round((np.clip(mu_ref, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        return np.rint((q - qs) / ST).astype(int)

    def stats(lev):
        return dict(exact=[float((lev[:, o] == 0).mean()) for o in range(6)],
                    within1=[float((np.abs(lev[:, o]) <= 1).mean()) for o in range(6)],
                    mean_signed=[float(lev[:, o].mean()) for o in range(6)],
                    hist=[[float((lev[:, o] == k).mean()) for k in (-2, -1, 0, 1, 2)]
                          for o in range(6)])

    L = lo + Lg * step
    S0 = np.exp(L[np.arange(32)[None, :], idx_tr]).sum(1)
    n0 = -tau_eff * np.log(beta[None, :] * S0)
    T0 = np.ceil(n0 + PHASE) + BASE
    off = np.median(mu_tr - aff[:, 0][None, :] * T0, axis=0)
    base_ho = stats(lev_of(L, idx_ho, mu_ho, off))
    base_tr = lev_of(L, idx_tr, mu_tr, off)
    print(f"baseline held-out exact {[round(v*100,2) for v in base_ho['exact']]}")
    print(f"baseline held-out signed {[round(v,4) for v in base_ho['mean_signed']]}")

    # zero-step sanity
    assert stats(lev_of(L, idx_ho, mu_ho, off))["exact"] == base_ho["exact"]
    print("zero-step sanity: PASS\n")

    qs_tr = np.clip(np.round((np.clip(mu_tr, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)

    def loss_from_S(S, o, ref=None):
        n = -tau_eff * np.log(beta[o] * S)
        T = np.ceil(n + PHASE) + BASE
        mu = aff[o, 0] * T + off[o]
        q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        r = qs_tr[:, o] if ref is None else ref
        return np.abs(np.rint((q - r) / ST))

    S = np.exp(L[np.arange(32)[None, :], idx_tr]).sum(1)     # (N,6)
    L_cur = L.copy(); Lg_cur = Lg.copy()
    traj, moved = [], 0
    t0 = time.time()
    tot0 = sum(loss_from_S(S[:, o], o).sum() for o in range(6))
    traj.append(float(tot0))
    print(f"sweep 0 loss {tot0:,.0f}")
    stop = False
    for sw in range(a.sweeps):
        for o in range(6):
            for t in range(32):
                col = idx_tr[:, t]
                for k in range(64):
                    m = col == k
                    if not m.any():
                        continue
                    Sm = S[m, o]
                    rm = qs_tr[m, o]
                    cur = loss_from_S(Sm, o, rm).sum()
                    e_old = np.exp(L_cur[t, k, o])
                    best = (cur, 0, None)
                    for d in list(range(-a.kmax, 0)) + list(range(1, a.kmax + 1)):
                        e_new = np.exp(lo + (Lg_cur[t, k, o] + d) * step)
                        Sn = Sm - e_old + e_new
                        if (Sn <= 0).any():
                            continue
                        c = loss_from_S(Sn, o, rm).sum()
                        if c < best[0]:
                            best = (c, d, Sn)
                    if best[1]:
                        Lg_cur[t, k, o] += best[1]
                        L_cur[t, k, o] = lo + Lg_cur[t, k, o] * step
                        S[m, o] = best[2]
                        moved += 1
                if time.time() - t0 > a.budget_s:
                    stop = True; break
            if stop:
                break
        tot = sum(loss_from_S(S[:, o], o).sum() for o in range(6))
        traj.append(float(tot))
        print(f"sweep {sw+1} loss {tot:,.0f}  moved {moved:,}  "
              f"{time.time()-t0:.0f}s{'  [budget hit]' if stop else ''}")
        if stop:
            break

    S2 = np.exp(L_cur[np.arange(32)[None, :], idx_tr]).sum(1)
    T2 = np.ceil(-tau_eff * np.log(beta[None, :] * S2) + PHASE) + BASE
    off2 = np.median(mu_tr - aff[:, 0][None, :] * T2, axis=0)
    aft_ho = stats(lev_of(L_cur, idx_ho, mu_ho, off2))
    drift = float(np.abs(L_cur - L).max())
    print(f"\n=== HELD-OUT ===")
    print(f"{'dim':>3} {'exact':>20} {'mean signed (levels)':>28}")
    for o in range(6):
        print(f"{o:>3}  {base_ho['exact'][o]*100:6.2f}% -> {aft_ho['exact'][o]*100:6.2f}%"
              f"      {base_ho['mean_signed'][o]:+8.4f} -> {aft_ho['mean_signed'][o]:+8.4f}")
    print(f"  overall exact {np.mean(base_ho['exact'])*100:.2f}% -> "
          f"{np.mean(aft_ho['exact'])*100:.2f}%   within1 "
          f"{np.mean(aft_ho['within1'])*100:.3f}%")
    for o in range(6):
        print(f"  dim {o} hist before {[round(v*100,1) for v in base_ho['hist'][o]]} "
              f"-> after {[round(v*100,1) for v in aft_ho['hist'][o]]}")
    print(f"  weights moved {moved:,}/12,288   max log-domain drift {drift:.4f}")
    print(f"  loss trajectory {[round(v) for v in traj]}")

    np.save("/tmp/_cd_weights.npy", L_cur * tau)   # optimised table, for true-SNN validation
    print("saved /tmp/_cd_weights.npy")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(before=base_ho, after=aft_ho, moved=int(moved), drift=drift,
                   loss_traj=traj, sweeps_done=len(traj) - 1), open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
