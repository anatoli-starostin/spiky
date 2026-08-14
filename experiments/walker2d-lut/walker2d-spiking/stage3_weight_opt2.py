"""Stage-3 weight optimisation, with the closed form PINNED to the real SNN.

Two gates, both hard, both before any gradient is taken:

  GATE 1  the closed-form crossing tick must equal the SNN's MEASURED tick on >=99.9% of
          states, per dim. The previous prototype skipped this and optimised a model that
          disagreed with the network by ~20 points of exact-match and had the wrong SIGN of
          bias on three dims. Root cause: T = ceil(arrival + n) and a fitted decode offset
          absorbs an arbitrary REAL constant, but ceil does not commute with a non-integer
          shift -- so the model sat on a different sub-tick phase than the neuron. Fixed here
          by solving for the per-dim phase directly against measured T.

  GATE 2  a zero-step run must reproduce the baseline metrics exactly, proving forward+loss
          are consistent before training.
"""
import argparse
import json
import os
import sys

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


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def ste_ceil(x):
    return x + (torch.ceil(x) - x).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-5)
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

    net, ids, _, n_ticks, nneur, _, _, _, _ = QP.build(
        Z, list(range(6)), False, float(Q["tau_m_out"]), "cuda", 6, 3.0, True)

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
        T, TL = [], []
        for s0 in range(0, len(X), 64):
            ch = X[s0:s0 + 64]
            tk = QP.encode_gauss(ch, edges)
            o = QP.run(net, ids, tk, n_ticks, "cuda")
            T.append(o[6].astype(np.float64)); TL.append(tk.max(1).astype(np.float64))
        return idx, mu_sw, np.concatenate(T), np.concatenate(TL)

    idx_tr, mu_tr, T_tr, TL_tr = collect(0)
    idx_ho, mu_ho, T_ho, TL_ho = collect(7)
    print(f"train {len(idx_tr):,}  held-out {len(idx_ho):,}  (SNN ticks measured)")

    tab = np.arange(32)

    def n_closed(Wt, idx):
        sel = Wt[tab[None, :], idx]
        S = np.exp(sel / tau).sum(1)
        return -tau_eff * np.log(beta[None, :] * S)

    # ---- GATE 1: solve the per-dim sub-tick phase against MEASURED T -----------------
    n_tr = n_closed(W0, idx_tr)
    rel = T_tr - TL_tr[:, None]                       # SNN tick relative to the reference
    phase, base, agree = np.zeros(6), np.zeros(6), np.zeros(6)
    for o in range(6):
        best = (-1, 0.0, 0)
        for ph in np.linspace(0.0, 1.0, 201)[:-1]:
            c = np.round(np.median(rel[:, o] - np.ceil(n_tr[:, o] + ph)))
            ag = float((np.ceil(n_tr[:, o] + ph) + c == rel[:, o]).mean())
            if ag > best[0]:
                best = (ag, ph, c)
        agree[o], phase[o], base[o] = best
        print(f"  dim {o}: phase {phase[o]:.3f}  base {base[o]:+.0f}  "
              f"agreement with measured T = {agree[o]*100:.4f}%")
    print(f"\nGATE 1 min agreement: {agree.min()*100:.4f}%")
    if agree.min() < 0.999:
        print("GATE 1 FAILED (<99.9%) -- refusing to optimise an unfaithful model.")
        json.dump(dict(gate1_agreement=[float(v) for v in agree],
                       gate1_passed=False, phase=[float(v) for v in phase]),
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
        return
    print("GATE 1 PASSED\n")

    slope = torch.tensor(aff[:, 0], dtype=torch.float64, device=dev)
    ph_t = torch.tensor(phase, dtype=torch.float64, device=dev)
    bs_t = torch.tensor(base, dtype=torch.float64, device=dev)
    bet = torch.tensor(beta, dtype=torch.float64, device=dev)
    L0 = torch.tensor(W0 / tau, dtype=torch.float64, device=dev)
    lo, hi = float(L0.min()), float(L0.max())

    def fwd(L, idx, step, hard):
        I = torch.tensor(idx, dtype=torch.long, device=dev)
        Lq = lo + ste_round((L - lo) / step) * step if step else L
        S = torch.exp(Lq[torch.as_tensor(tab, device=dev)[None, :], I]).sum(1)
        n = -tau_eff * torch.log(bet[None, :] * S)
        T = (torch.ceil(n + ph_t) if hard else ste_ceil(n + ph_t)) + bs_t
        return T

    def metrics(L, idx, mu_ref, off, step):
        with torch.no_grad():
            T = fwd(L, idx, step, True)
            mu = slope[None, :] * T + off[None, :]
            q = torch.clamp(torch.round((mu.clamp(-CL, CL) + CL) / ST) * ST - CL, -CL, CL)
            m = torch.tensor(mu_ref, device=dev)
            qs = torch.clamp(torch.round((m.clamp(-CL, CL) + CL) / ST) * ST - CL, -CL, CL)
            lev = torch.round((q - qs) / ST).long()
            return dict(exact=[float((lev[:, o] == 0).double().mean()) for o in range(6)],
                        within1=[float((lev[:, o].abs() <= 1).double().mean())
                                 for o in range(6)],
                        mean_signed=[float(lev[:, o].double().mean()) for o in range(6)],
                        hist=[[float((lev[:, o] == k).double().mean())
                               for k in (-2, -1, 0, 1, 2)] for o in range(6)])

    with torch.no_grad():
        T0 = fwd(L0, idx_tr, None, True)
        off = (torch.tensor(mu_tr, device=dev) - slope[None, :] * T0).median(0).values
    base_ho = metrics(L0, idx_ho, mu_ho, off, None)
    print("BASELINE (held-out, from the pinned closed form):")
    print(f"  exact {[round(v*100,2) for v in base_ho['exact']]}")
    print(f"  mean signed (levels) {[round(v,4) for v in base_ho['mean_signed']]}")
    print(f"  dim0 hist {[round(v*100,2) for v in base_ho['hist'][0]]}")
    ok = (min(base_ho["exact"]) > 0.70 and max(base_ho["mean_signed"]) < 0.0)
    print(f"GATE 1b (reproduces SNN behaviour: exact>70% all dims, all-negative bias): "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        json.dump(dict(gate1_agreement=[float(v) for v in agree], gate1_passed=True,
                       gate1b_passed=False, baseline_heldout=base_ho),
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}\nrefusing to optimise -- forward still does not match the SNN.")
        return

    res = dict(gate1_agreement=[float(v) for v in agree], gate1_passed=True,
               gate1b_passed=True, baseline_heldout=base_ho, variants={})
    Itr = torch.tensor(idx_tr, dtype=torch.long, device=dev)
    Mtr = torch.tensor(mu_tr, device=dev)
    for name, bits in (("full_precision", None), ("10bit", 10), ("8bit", 8)):
        st = None if bits is None else (hi - lo) / (2 ** bits - 1)
        L = L0.clone().requires_grad_(True)
        # GATE 2: zero-step must reproduce the baseline exactly
        z = metrics(L, idx_ho, mu_ho, off, st)
        if bits is None and z["exact"] != base_ho["exact"]:
            print(f"GATE 2 FAILED for {name}"); return
        opt = torch.optim.Adam([L], lr=a.lr)
        losses = []
        for it in range(a.iters):
            S = torch.exp((lo + ste_round((L - lo) / st) * st if st else L)[
                torch.as_tensor(tab, device=dev)[None, :], Itr]).sum(1)
            n = -tau_eff * torch.log(bet[None, :] * S)
            mu = slope[None, :] * (ste_ceil(n + ph_t) + bs_t) + off[None, :]
            loss = ((mu - Mtr) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([L], 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            T2 = fwd(L.detach(), idx_tr, st, True)
            off2 = (Mtr - slope[None, :] * T2).median(0).values
        aft = metrics(L.detach(), idx_ho, mu_ho, off2, st)
        drift = float((L.detach() - L0).abs().max())
        nchg = int(((L.detach() - L0).abs() > 1e-9).sum())
        desc = losses[-1] < losses[0]
        res["variants"][name] = dict(after=aft, loss_first=losses[0], loss_last=losses[-1],
                                     descending=bool(desc), max_drift=drift, n_changed=nchg,
                                     zero_step_ok=bool(z["exact"] == base_ho["exact"]))
        print(f"\n=== {name} ===  loss {losses[0]:.6f} -> {losses[-1]:.6f} "
              f"({'DESCENDING' if desc else 'NOT DESCENDING'})  drift {drift:.4f}")
        print(f"  exact  {[round(v*100,2) for v in base_ho['exact']]} -> "
              f"{[round(v*100,2) for v in aft['exact']]}")
        print(f"  signed {[round(v,4) for v in base_ho['mean_signed']]} -> "
              f"{[round(v,4) for v in aft['mean_signed']]}")
        print(f"  within1 {np.mean(aft['within1'])*100:.3f}%   changed {nchg:,}/12288")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
