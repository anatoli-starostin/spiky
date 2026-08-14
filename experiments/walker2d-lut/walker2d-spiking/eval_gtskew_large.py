"""Large paired eval: GT-skew spiking build vs the software quantised policy.

Both run under the IDENTICAL harness, env count and seed, so the comparison is against a
freshly measured baseline rather than a remembered number.

Note on "paired per-episode": the two policies emit different actions from the first step, so
trajectories diverge immediately and episodes cannot be matched beyond their shared initial
state. Reporting a per-episode paired difference would be fiction. The shared seed still
removes initial-state variance, which is the honest part of the pairing.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Everything resolves relative to this file, the way tiny_lut_quantised_pipeline.py does,
# so a clone or a worktree anywhere runs against its own tree. `warp_env` genuinely lives
# in the training tree next door; the pipeline module is this script's own neighbour.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from warp_env import WarpWalker2dVecEnv                     # noqa: E402
import tiny_lut_quantised_pipeline as QP                    # noqa: E402

NPZ = os.path.join(HERE, "..", "exp19_lut-lse-expmlpcrit-t32", "deploy", "quantised",
                   "walker2d_fastlut_lse_exp19_quantised.npz")
ACT = os.path.join(HERE, "deploy_quantised", "spiking_lut_quantised_actor.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", default="0,7")
    ap.add_argument("--weights", default=None,
                    help="optimised Stage-3 table (.npy); default = shipped")
    ap.add_argument("--offset", default=None,
                    help="decode offset fitted WITH those weights (.npy). Required whenever --weights is given: the fit re-derives the offset, and using the shipped one instead mis-decodes by ~0.25 level.")
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
    W = Z["weights"].astype(np.float64)
    aff = Q["affine"].astype(np.float64)
    omt = torch.tensor(Z["obs_mean"], dtype=torch.float32, device=dev)
    ovt = torch.tensor(Z["obs_var"], dtype=torch.float32, device=dev)

    class NpzView:
        """Z with the Stage-3 table swapped; QP.build consumes it unchanged."""
        def __init__(self, b, W):
            self._b, self._W, self.files = b, W, list(b.files)

        def __getitem__(self, k):
            return self._W if k == "weights" else self._b[k]

    Zb = Z
    if a.weights:
        Wo = np.load(a.weights).astype(np.float64)
        d = np.abs(Wo - Z["weights"].astype(np.float64))
        print(f"optimised table: {int((d > 1e-12).sum())} of {d.size} weights differ, "
              f"max |dw| {d.max():.6f}")
        Zb = NpzView(Z, Wo)
        if a.offset is None:
            raise SystemExit("--weights given without --offset: the coordinate-descent fit "
                             "re-derives the decode offset, and decoding optimised weights "
                             "with the shipped offset mis-decodes by ~0.25 level. Refusing.")
        aff = aff.copy()
        aff[:, 1] = np.load(a.offset).astype(np.float64)
        print(f"decode offset: using the fitted one "
              f"{[round(float(v), 5) for v in aff[:, 1]]}")
    net, ids, nsyn, n_ticks, nneur, _, _, _, dmax = QP.build(
        Zb, list(range(6)), False, float(Q["tau_m_out"]), "cuda", 6, 3.0, True)
    print(f"SNN: {nneur} neurons, {nsyn} synapses, dmax {dmax}, n_ticks {n_ticks}")

    def act_sw(x):
        g = np.searchsorted(edges, x.ravel(), side="left").reshape(x.shape)
        xq = dq[np.clip(g, 0, len(dq) - 1)]
        idx = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)
        sel = W[np.arange(32)[None, :], idx]
        mu = 32 * tau * np.log(np.exp(sel / tau).mean(1))
        return np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)

    def act_snn(x):
        tk = QP.encode_gauss(x, edges)
        o = QP.run(net, ids, tk, n_ticks, "cuda")
        T = o[6].astype(np.float64)
        mu = np.where(T < n_ticks,
                      aff[:, 0][None, :] * (T - tk.max(1)[:, None]) + aff[:, 1][None, :], -CL)
        return np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)

    def rollout(which, seed):
        env = WarpWalker2dVecEnv(num_envs=a.envs, seed=seed, solver_iters=100,
                                 ls_iters=50, obs_clip_vel=10.0)
        obs = env.reset()
        ep = torch.zeros(env.N, device=dev); el = torch.zeros(env.N, device=dev)
        rets, lens = [], []
        for _ in range(a.steps):
            x = ((obs - omt) / torch.sqrt(ovt + 1e-8)).cpu().numpy().astype(np.float64)
            q = act_sw(x) if which == "sw" else act_snn(x)
            obs, rew, term, trunc = env.step(torch.as_tensor(q, dtype=torch.float32,
                                                             device=dev))
            ep += rew; el += 1
            d = term | trunc
            if d.any():
                rets.append(ep[d].clone()); lens.append(el[d].clone())
                ep = ep * (~d).float(); el = el * (~d).float()
        return (torch.cat(rets).cpu().numpy() if rets else np.zeros(0),
                torch.cat(lens).cpu().numpy() if lens else np.zeros(0))

    seeds = [int(s) for s in a.seeds.split(",")]
    res = {}
    for which in ("sw", "snn"):
        R, L = [], []
        for sd in seeds:
            t0 = time.time()
            r, l = rollout(which, sd)
            R.append(r); L.append(l)
            print(f"  {which} seed {sd}: {len(r)} eps, mean {r.mean():.1f}, "
                  f"{time.time()-t0:.0f}s")
        R, L = np.concatenate(R), np.concatenate(L)
        res[which] = dict(n=int(len(R)), mean=float(R.mean()), std=float(R.std()),
                          se=float(R.std() / np.sqrt(len(R))), median=float(np.median(R)),
                          mean_len=float(L.mean()))
        print(f"{which:>4}: n={len(R):4d}  mean {R.mean():7.1f} +- {R.std():6.1f}  "
              f"se {R.std()/np.sqrt(len(R)):5.1f}  median {np.median(R):7.1f}  "
              f"len {L.mean():.0f}")
    d = res["snn"]["mean"] - res["sw"]["mean"]
    se = float(np.hypot(res["snn"]["se"], res["sw"]["se"]))
    res["diff"] = dict(value=float(d), se=se, sigma=float(d / se),
                       distinguishable=bool(abs(d) > 2 * se))
    print(f"\nSNN - software = {d:+.1f} +- {se:.1f}  ({d/se:+.2f} sigma)  "
          f"{'DISTINGUISHABLE' if abs(d) > 2*se else 'not distinguishable'} at 2 se")
    res["build"] = dict(neurons=int(nneur), synapses=int(nsyn), n_ticks=int(n_ticks))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
