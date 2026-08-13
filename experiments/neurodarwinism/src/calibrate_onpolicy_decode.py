"""Re-fit the spiking decode OFFSET on ON-POLICY states, centring the mean signed residual.

Why on-policy: the offset was fitted on the distill-pool held-out split, and the resulting
constant is ~0.2 level wrong on the states the policy actually visits -- which showed up as
100% one-sided -1 errors at ~21% (a near-constant negative offset, not rounding and not
symmetric jitter).

Why bias-centring rather than exact-match: maximising exact-match re-selects the biased-low
solution (verified -- it returned a 0.0 shift while the bias sat at -0.207 levels). The
objective that matters for return is a zero mean signed residual.

Calibrates on one rollout, VERIFIES on a second independent rollout (different seed), so the
reported after-numbers are not the ones that were fitted.
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
    ap.add_argument("--write", action="store_true", help="write corrected offsets to the npz")
    a = ap.parse_args()
    dev = torch.device("cuda")

    Z, Q = np.load(NPZ), np.load(ACT)
    tau = float(Z["tau_actor"])
    CL, LV = float(Z["out_quant_clip"]), int(Z["out_quant_levels"])
    ST = 2 * CL / (LV - 1)
    om, ov = Z["obs_mean"], Z["obs_var"]
    edges, dq = Z["in_quant_edges"], Z["in_quant_dequant"]
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    pw = 1 << np.arange(A_.shape[1] - 1, -1, -1)
    W = Z["weights"].astype(np.float64)
    aff = Q["affine"].astype(np.float64)

    def software(xn):
        g = np.searchsorted(edges, xn.ravel(), side="left").reshape(xn.shape)
        xq = dq[np.clip(g, 0, len(dq) - 1)]
        idx = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)
        sel = W[np.arange(32)[None, :], idx]
        return 32 * tau * np.log(np.exp(sel / tau).mean(1))

    net, ids, nsyn, n_ticks, nneur, _, _, _, dmax = QP.build(
        Z, list(range(6)), False, float(Q["tau_m_out"]), "cuda", 6, 3.0, True)
    print(f"SNN: {nneur} neurons, {nsyn} synapses, dmax {dmax}, n_ticks {n_ticks}")

    def collect(seed):
        env = WarpWalker2dVecEnv(num_envs=a.envs, seed=seed, solver_iters=100, ls_iters=50,
                                 obs_clip_vel=10.0)
        obs = env.reset()
        Xs = []
        omt = torch.tensor(om, dtype=torch.float32, device=dev)
        ovt = torch.tensor(ov, dtype=torch.float32, device=dev)
        for _ in range(a.steps):
            x = ((obs - omt) / torch.sqrt(ovt + 1e-8)).cpu().numpy().astype(np.float64)
            Xs.append(x)
            mu = software(x)
            q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
            obs, _, _, _ = env.step(torch.as_tensor(q, dtype=torch.float32, device=dev))
        X = np.concatenate(Xs)
        T, TL = [], []
        for s in range(0, len(X), 64):
            ch = X[s:s + 64]
            tk = QP.encode_gauss(ch, edges)
            o = QP.run(net, ids, tk, n_ticks, "cuda")
            T.append(o[6].astype(np.float64)); TL.append(tk.max(1).astype(np.float64))
        return X, np.concatenate(T), np.concatenate(TL), software(X)

    def stats(T, TL, mu_sw, off):
        live = T < n_ticks
        mu = np.where(live, aff[:, 0][None, :] * (T - TL[:, None]) + off[None, :], -CL)
        q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        qs = np.clip(np.round((np.clip(mu_sw, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        lev = np.rint((q - qs) / ST).astype(int)
        return dict(
            mean_signed=[float((q[:, o] - qs[:, o]).mean()) for o in range(6)],
            mean_signed_levels=[float(lev[:, o].mean()) for o in range(6)],
            exact=[float((lev[:, o] == 0).mean()) for o in range(6)],
            within1=[float((np.abs(lev[:, o]) <= 1).mean()) for o in range(6)],
            hist=[[float((lev[:, o] == k).mean()) for k in (-2, -1, 0, 1, 2)]
                  for o in range(6)])

    # ---- calibrate on rollout A -------------------------------------------------------
    _, Ta, TLa, mua = collect(0)
    before = stats(Ta, TLa, mua, aff[:, 1])
    corr = -np.array(before["mean_signed"])
    off2 = aff[:, 1] + corr
    print(f"\nper-dim correction (level units): "
          f"{[round(c/ST, 4) for c in corr]}")

    # ---- verify on an INDEPENDENT rollout B -------------------------------------------
    _, Tb, TLb, mub = collect(7)
    b_before = stats(Tb, TLb, mub, aff[:, 1])
    b_after = stats(Tb, TLb, mub, off2)

    print(f"\nHELD-OUT rollout (seed 7), before -> after")
    print(f"{'dim':>3} {'mean signed (levels)':>26} {'exact':>16} {'within1':>16}")
    for o in range(6):
        print(f"{o:>3} {b_before['mean_signed_levels'][o]:+11.4f} ->"
              f"{b_after['mean_signed_levels'][o]:+11.4f}  "
              f"{b_before['exact'][o]*100:6.2f}% ->{b_after['exact'][o]*100:6.2f}%  "
              f"{b_before['within1'][o]*100:6.2f}% ->{b_after['within1'][o]*100:6.2f}%")
    print(f"\nhistogram after (-2,-1,0,+1,+2), per dim:")
    for o in range(6):
        print(f"  dim {o}: " + " ".join(f"{v*100:6.2f}" for v in b_after["hist"][o]))
    ms_b = np.mean(b_before["mean_signed_levels"]); ms_a = np.mean(b_after["mean_signed_levels"])
    print(f"\nmean signed overall: {ms_b:+.4f} -> {ms_a:+.4f} levels")

    res = dict(neurons=int(nneur), synapses=int(nsyn), dmax=int(dmax),
               correction_levels=[float(c / ST) for c in corr],
               correction_abs=[float(c) for c in corr],
               heldout_before=b_before, heldout_after=b_after)
    if a.write:
        D = dict(np.load(ACT)); aff2 = aff.copy(); aff2[:, 1] = off2
        D["affine"] = aff2
        np.savez_compressed(ACT, **D)
        print(f"\nwrote corrected offsets into {ACT}")
        res["written"] = True
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
