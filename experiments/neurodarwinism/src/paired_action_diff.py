"""Paired action comparison on ON-POLICY states: software quantised policy vs the GT-skew SNN.

The parity gate is measured on the distill pool. This measures the states the policy actually
visits, and reports the MEAN SIGNED residual -- symmetric jitter averages to ~0, a decode
offset bias does not.
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
ACT = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
       "exp012_tiny-direct-genome/deploy_quantised/spiking_lut_quantised_actor.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")

    Z = np.load(NPZ)
    Q = np.load(ACT)
    tau = float(Z["tau_actor"])
    CL, LV = float(Z["out_quant_clip"]), int(Z["out_quant_levels"])
    ST = 2 * CL / (LV - 1)
    om, ov = Z["obs_mean"], Z["obs_var"]
    edges, dq = Z["in_quant_edges"], Z["in_quant_dequant"]
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    pw = 1 << np.arange(A_.shape[1] - 1, -1, -1)
    W = Z["weights"].astype(np.float64)

    def software(xn):
        g = np.searchsorted(edges, xn.ravel(), side="left").reshape(xn.shape)
        xq = dq[np.clip(g, 0, len(dq) - 1)]
        idx = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)
        sel = W[np.arange(32)[None, :], idx]
        mu = 32 * tau * np.log(np.exp(sel / tau).mean(1))
        return np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL), mu

    # ---- 1. on-policy states from the SOFTWARE policy -------------------------------
    env = WarpWalker2dVecEnv(num_envs=a.envs, seed=0, solver_iters=100, ls_iters=50,
                             obs_clip_vel=10.0)
    obs = env.reset()
    OBS = []
    for _ in range(a.steps):
        xn = ((obs - torch.tensor(om, dtype=torch.float32, device=dev))
              / torch.sqrt(torch.tensor(ov, dtype=torch.float32, device=dev) + 1e-8))
        x = xn.cpu().numpy().astype(np.float64)
        OBS.append(x)
        q, _ = software(x)
        obs, _, _, _ = env.step(torch.as_tensor(q, dtype=torch.float32, device=dev))
    X = np.concatenate(OBS)
    print(f"on-policy observations collected: {X.shape[0]:,}")

    q_sw, mu_sw = software(X)

    # ---- 2. the same states through the GT-skew SNN ---------------------------------
    net, ids, nsyn, n_ticks, nneur, _, _, _, dmax = QP.build(
        Z, list(range(6)), False, float(Q["tau_m_out"]), "cuda", 6, 3.0, True)
    print(f"SNN: {nneur} neurons, {nsyn} synapses, n_ticks {n_ticks}")
    aff = Q["affine"].astype(np.float64)
    T, TL = [], []
    for s in range(0, len(X), 64):
        ch = X[s:s + 64]
        tk = QP.encode_gauss(ch, edges)
        o = QP.run(net, ids, tk, n_ticks, "cuda")
        T.append(o[6].astype(np.float64)); TL.append(tk.max(1).astype(np.float64))
    T, TL = np.concatenate(T), np.concatenate(TL)
    live = T < n_ticks

    def decode(off):
        mu = aff[:, 0][None, :] * (T - TL[:, None]) + off[None, :]
        mu = np.where(live, mu, -CL)
        return np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)

    q_sn = decode(aff[:, 1])

    # ---- 3. residual statistics -----------------------------------------------------
    lev = np.rint((q_sn - q_sw) / ST).astype(int)
    res = dict(n=int(len(X)), neurons=int(nneur), synapses=int(nsyn),
               exact_overall=float((lev == 0).mean()),
               within1_overall=float((np.abs(lev) <= 1).mean()), per_dim=[])
    print(f"\n{'dim':>3} {'exact':>8} {'within1':>8} {'MEAN SIGNED':>13} {'mean|res|':>10} "
          f"{'-2':>6} {'-1':>7} {'0':>7} {'+1':>7} {'+2':>6}")
    for o in range(6):
        h = [float((lev[:, o] == k).mean()) for k in (-2, -1, 0, 1, 2)]
        d = dict(dim=o, exact=float((lev[:, o] == 0).mean()),
                 within1=float((np.abs(lev[:, o]) <= 1).mean()),
                 mean_signed=float((q_sn[:, o] - q_sw[:, o]).mean()),
                 mean_signed_levels=float(lev[:, o].mean()),
                 mean_abs=float(np.abs(q_sn[:, o] - q_sw[:, o]).mean()), hist=h)
        res["per_dim"].append(d)
        print(f"{o:>3} {d['exact']*100:7.2f}% {d['within1']*100:7.2f}% "
              f"{d['mean_signed']:+13.5f} {d['mean_abs']:10.5f} "
              + " ".join(f"{v*100:6.2f}" for v in h))
    ms = np.array([d["mean_signed"] for d in res["per_dim"]])
    print(f"\noverall exact {res['exact_overall']*100:.2f}%  "
          f"within1 {res['within1_overall']*100:.2f}%  "
          f"mean signed {ms.mean():+.5f} (level units {ms.mean()/ST:+.3f})")

    # ---- 4. re-fit the offset on THIS on-policy set ----------------------------------
    off2 = aff[:, 1].copy()
    for o in range(6):
        cand = off2[o] + np.linspace(-ST, ST, 161)
        sc = [( np.abs(np.clip(np.round((np.clip(np.where(live[:, o],
                aff[o, 0] * (T[:, o] - TL) + c, -CL), -CL, CL) + CL) / ST) * ST - CL,
                -CL, CL) - q_sw[:, o]) < 1e-9).mean() for c in cand]
        off2[o] = float(cand[int(np.argmax(sc))])
    q2 = decode(off2)
    lev2 = np.rint((q2 - q_sw) / ST).astype(int)
    res["refit"] = dict(offset_delta=[float(off2[o] - aff[o, 1]) for o in range(6)],
                        exact=float((lev2 == 0).mean()),
                        mean_signed=[float((q2[:, o] - q_sw[:, o]).mean()) for o in range(6)])
    print(f"\nAFTER re-fitting the offset on the on-policy set:")
    print(f"  exact {res['exact_overall']*100:.2f}% -> {res['refit']['exact']*100:.2f}%")
    print(f"  mean signed per dim -> "
          f"{[round(v, 5) for v in res['refit']['mean_signed']]}")
    print(f"  offset shift (level units) "
          f"{[round(v/ST, 3) for v in res['refit']['offset_delta']]}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
