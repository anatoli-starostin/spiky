"""Treat the Stage-3 readout weights as an optimisation problem, with the tick-ceil and a
10-bit log-domain weight grid both inside the loss via straight-through estimators.

No SNN in the loop: the amplitude readout has an exact closed form. With arrivals coincident
on one tick and a linear anti-leak membrane,

    S = sum_t exp(w_sel[t,o]/tau)          (over the 32 selected cells)
    n = -tau_eff * log(beta_o * S)          ticks-after-arrival to threshold
    T = ceil(n)                             the engine emits on integer ticks
    mu = slope*(T - ref) + offset           the shipped affine decode
    q  = snap(clip(mu))                     the 22-level output grid

Stage-1/2 selection is FROZEN -- the selected cell indices are computed once from the
software path and never change, so address-bit parity stays bit-exact by construction and
only the 12,288 table values are free.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/walker2d-lut/src")
from warp_env import WarpWalker2dVecEnv                     # noqa: E402

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
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bits", type=int, default=10)
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

    def rollout(seed):
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
            idx = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)
            sel = W0[np.arange(32)[None, :], idx]
            mu = 32 * tau * np.log(np.exp(sel / tau).mean(1))
            q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
            obs, _, _, _ = env.step(torch.as_tensor(q, dtype=torch.float32, device=dev))
        X = np.concatenate(Xs)
        g = np.searchsorted(edges, X.ravel(), side="left").reshape(X.shape)
        xq = dq[np.clip(g, 0, len(dq) - 1)]
        idx = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)   # (N,32) FROZEN
        sel = W0[np.arange(32)[None, :], idx]
        mu_sw = 32 * tau * np.log(np.exp(sel / tau).mean(1))
        return idx, mu_sw

    idx_tr, mu_tr = rollout(0)
    idx_ho, mu_ho = rollout(7)
    print(f"train states {len(idx_tr):,}   held-out states {len(idx_ho):,}")

    tab = np.arange(32)
    L0 = torch.tensor(W0 / tau, dtype=torch.float64, device=dev)      # log-domain
    lo, hi = float(L0.min()), float(L0.max())
    step = (hi - lo) / (2 ** a.bits - 1)
    print(f"log-domain range [{lo:.3f}, {hi:.3f}]  {a.bits}-bit step {step:.6f}")

    slope = torch.tensor(aff[:, 0], dtype=torch.float64, device=dev)
    offs = torch.tensor(aff[:, 1], dtype=torch.float64, device=dev)
    bet = torch.tensor(beta, dtype=torch.float64, device=dev)

    def forward(L, idx, quant):
        I = torch.tensor(idx, dtype=torch.long, device=dev)
        Lq = lo + ste_round((L - lo) / step) * step if quant else L
        sel = Lq[torch.as_tensor(tab, device=dev)[None, :], I]        # (N,32,6)
        S = torch.exp(sel).sum(1)                                     # (N,6)
        n = -tau_eff * torch.log(bet[None, :] * S)
        T = ste_ceil(n)
        return slope[None, :] * T + offs[None, :] + slope[None, :] * 0.0, n

    # the shipped decode is mu = slope*(T - ref) + off; ref is absorbed into `off` below
    def fit_ref(L, idx, mu_ref, quant):
        with torch.no_grad():
            _, n = forward(L, idx, quant)
            T = torch.ceil(n)
            m = torch.tensor(mu_ref, device=dev)
            return (m - slope[None, :] * T).median(0).values

    def evaluate(L, idx, mu_ref, off, quant):
        with torch.no_grad():
            _, n = forward(L, idx, quant)
            mu = slope[None, :] * torch.ceil(n) + off[None, :]
            q = torch.clamp(torch.round((mu.clamp(-CL, CL) + CL) / ST) * ST - CL, -CL, CL)
            m = torch.tensor(mu_ref, device=dev)
            qs = torch.clamp(torch.round((m.clamp(-CL, CL) + CL) / ST) * ST - CL, -CL, CL)
            lev = torch.round((q - qs) / ST).long()
            return dict(
                exact=[float((lev[:, o] == 0).double().mean()) for o in range(6)],
                within1=[float((lev[:, o].abs() <= 1).double().mean()) for o in range(6)],
                mean_signed_levels=[float(lev[:, o].double().mean()) for o in range(6)],
                hist=[[float((lev[:, o] == k).double().mean()) for k in (-2, -1, 0, 1, 2)]
                      for o in range(6)])

    res = {}
    for name, quant in (("full_precision", False), (f"{a.bits}bit", True)):
        L = L0.clone().requires_grad_(True)
        off = fit_ref(L0, idx_tr, mu_tr, quant)
        base_ho = evaluate(L0, idx_ho, mu_ho, off, quant)
        opt = torch.optim.Adam([L], lr=a.lr)
        Itr = torch.tensor(idx_tr, dtype=torch.long, device=dev)
        Mtr = torch.tensor(mu_tr, device=dev)
        for it in range(a.iters):
            Lq = lo + ste_round((L - lo) / step) * step if quant else L
            sel = Lq[torch.as_tensor(tab, device=dev)[None, :], Itr]
            S = torch.exp(sel).sum(1)
            n = -tau_eff * torch.log(bet[None, :] * S)
            mu = slope[None, :] * ste_ceil(n) + off[None, :]
            loss = ((mu - Mtr) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 200 == 0:
                print(f"  [{name}] it {it:4d} loss {float(loss):.6f}")
        off2 = fit_ref(L.detach(), idx_tr, mu_tr, quant)
        after_ho = evaluate(L.detach(), idx_ho, mu_ho, off2, quant)
        drift = float((L.detach() - L0).abs().max())
        res[name] = dict(before=base_ho, after=after_ho, max_logdomain_drift=drift)
        print(f"\n=== {name} (HELD-OUT) ===")
        print(f"{'dim':>3} {'exact before->after':>26} {'mean signed (levels)':>28}")
        for o in range(6):
            print(f"{o:>3}  {base_ho['exact'][o]*100:6.2f}% -> {after_ho['exact'][o]*100:6.2f}%"
                  f"      {base_ho['mean_signed_levels'][o]:+8.4f} -> "
                  f"{after_ho['mean_signed_levels'][o]:+8.4f}")
        print(f"  overall exact {np.mean(base_ho['exact'])*100:.2f}% -> "
              f"{np.mean(after_ho['exact'])*100:.2f}%   "
              f"mean signed {np.mean(base_ho['mean_signed_levels']):+.4f} -> "
              f"{np.mean(after_ho['mean_signed_levels']):+.4f}   "
              f"within1 {np.mean(after_ho['within1'])*100:.2f}%")
        print(f"  hist after (-2,-1,0,+1,+2) dim0: "
              + " ".join(f"{v*100:6.2f}" for v in after_ho['hist'][0]))
        print(f"  max log-domain weight drift: {drift:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
