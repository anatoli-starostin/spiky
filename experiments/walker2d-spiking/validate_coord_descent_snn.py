"""Validate the coordinate-descent weights on the TRUE SNN (GPU), same held-out states.

The closed-form scorer disagrees with the network on ~0.2-2.3% of states, concentrated at
marginal ticks -- which is exactly where the optimiser works. So the closed-form 97.4% is a
screen, not a result. This rebuilds the network with the optimised weights and measures the
per-dim table from REAL spike ticks.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/walker2d-lut/src")
from warp_env import WarpWalker2dVecEnv                     # noqa: E402
import tiny_lut_quantised_pipeline as QP                    # noqa: E402
import stage3_coord_descent as CD                           # noqa: E402

NPZ = CD.NPZ
ACT = CD.ACT


class NpzView:
    """Z with the weights swapped -- lets QP.build consume optimised tables unchanged."""
    def __init__(self, base, W):
        self._b, self._W = base, W
        self.files = list(base.files)

    def __getitem__(self, k):
        return self._W if k == "weights" else self._b[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"

    Z, Q = np.load(NPZ), np.load(ACT)
    tau = float(Z["tau_actor"])
    CL, LV = float(Z["out_quant_clip"]), int(Z["out_quant_levels"])
    ST = 2 * CL / (LV - 1)
    aff = Q["affine"].astype(np.float64)
    edges = Z["in_quant_edges"]

    # ---- re-run the coordinate descent to obtain the optimised table -----------------
    sys.argv = ["cd", "--envs", str(a.envs), "--steps", str(a.steps),
                "--sweeps", str(a.sweeps), "--kmax", "3", "--budget-s", "300",
                "--out", "/tmp/_cd.json", "--return-weights", "1"]
    print("running coordinate descent to regenerate the optimised table ...")
    W_opt = np.load("/tmp/_cd_weights.npy") if os.path.exists("/tmp/_cd_weights.npy") else None
    if W_opt is None:
        print("ERROR: /tmp/_cd_weights.npy not found -- run stage3_coord_descent.py with "
              "--save-weights first.")
        return

    def collect(seed, net, ids, n_ticks):
        env = WarpWalker2dVecEnv(num_envs=a.envs, seed=seed, solver_iters=100,
                                 ls_iters=50, obs_clip_vel=10.0)
        obs = env.reset()
        omt = torch.tensor(Z["obs_mean"], dtype=torch.float32, device=dev)
        ovt = torch.tensor(Z["obs_var"], dtype=torch.float32, device=dev)
        A_, B_ = Z["anchor_a"], Z["anchor_b"]
        pw = 1 << np.arange(A_.shape[1] - 1, -1, -1)
        dq = Z["in_quant_dequant"]
        W0 = Z["weights"].astype(np.float64)
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
        i = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)
        s = W0[np.arange(32)[None, :], i]
        mu_sw = 32 * tau * np.log(np.exp(s / tau).mean(1))
        T, TL = [], []
        for s0 in range(0, len(X), 64):
            tk = QP.encode_gauss(X[s0:s0 + 64], edges)
            o = QP.run(net, ids, tk, n_ticks, dev)
            T.append(o[6].astype(np.float64)); TL.append(tk.max(1).astype(np.float64))
        return np.concatenate(T), np.concatenate(TL), mu_sw

    def metrics(T, TL, mu_sw, off):
        mu = aff[:, 0][None, :] * (T - TL[:, None]) + off[None, :]
        q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        qs = np.clip(np.round((np.clip(mu_sw, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        lev = np.rint((q - qs) / ST).astype(int)
        return dict(exact=[float((lev[:, o] == 0).mean()) for o in range(6)],
                    within1=[float((np.abs(lev[:, o]) <= 1).mean()) for o in range(6)],
                    mean_signed=[float(lev[:, o].mean()) for o in range(6)],
                    hist=[[float((lev[:, o] == k).mean()) for k in (-2, -1, 0, 1, 2)]
                          for o in range(6)])

    res = {}
    for name, Wuse in (("baseline", Z["weights"].astype(np.float64)), ("optimised", W_opt)):
        net, ids, nsyn, n_ticks, nneur, _, _, _, _ = QP.build(
            NpzView(Z, Wuse), list(range(6)), False, float(Q["tau_m_out"]), dev, 6, 3.0, True)
        Ttr, TLtr, mutr = collect(0, net, ids, n_ticks)
        off = np.median(mutr - aff[:, 0][None, :] * (Ttr - TLtr[:, None]), axis=0)
        Tho, TLho, muho = collect(7, net, ids, n_ticks)
        m = metrics(Tho, TLho, muho, off)
        res[name] = m
        print(f"\n=== TRUE SNN, {name} (held-out, GPU) ===  "
              f"{nneur} neurons {nsyn} synapses")
        print(f"  exact  {[round(v*100,2) for v in m['exact']]}  "
              f"overall {np.mean(m['exact'])*100:.2f}%")
        print(f"  signed {[round(v,4) for v in m['mean_signed']]}")
        print(f"  within1 {np.mean(m['within1'])*100:.3f}%")
        for o in range(6):
            print(f"    dim {o} hist {[round(v*100,1) for v in m['hist'][o]]}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
