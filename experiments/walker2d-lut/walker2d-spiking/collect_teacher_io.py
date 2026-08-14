"""Collect a large input->output dataset from the SOFTWARE teacher (shipped table W0).

⚠️ These are SOFTWARE-TEACHER-VISITED states -- the open-loop distribution. The spiking
policy's own closed-loop distribution is NOT sampled here, which is the same caveat that
explains why 97% action agreement on this kind of data moved closed-loop return by zero.
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

NPZ = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/"
       "exp19_lut-lse-expmlpcrit-t32/deploy/quantised/"
       "walker2d_fastlut_lse_exp19_quantised.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")

    Z = np.load(NPZ)
    tau = float(Z["tau_actor"])
    CL, LV = float(Z["out_quant_clip"]), int(Z["out_quant_levels"])
    ST = 2 * CL / (LV - 1)
    edges, dq = Z["in_quant_edges"], Z["in_quant_dequant"]
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    pw = 1 << np.arange(A_.shape[1] - 1, -1, -1)
    W0 = Z["weights"].astype(np.float64)
    om, ov = Z["obs_mean"], Z["obs_var"]
    omt = torch.tensor(om, dtype=torch.float32, device=dev)
    ovt = torch.tensor(ov, dtype=torch.float32, device=dev)

    RAW, NRM, TCK, MU, Q = [], [], [], [], []
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()
    for sd in seeds:
        env = WarpWalker2dVecEnv(num_envs=a.envs, seed=sd, solver_iters=100,
                                 ls_iters=50, obs_clip_vel=10.0)
        obs = env.reset()
        for _ in range(a.steps):
            raw = obs.cpu().numpy().astype(np.float32)
            xn = ((obs - omt) / torch.sqrt(ovt + 1e-8)).cpu().numpy().astype(np.float64)
            g = np.searchsorted(edges, xn.ravel(), side="left").reshape(xn.shape)
            g = np.clip(g, 0, len(dq) - 1)
            xq = dq[g]
            idx = (((xq[:, A_] - xq[:, B_]) > 0).astype(np.int64) * pw).sum(-1)
            sel = W0[np.arange(32)[None, :], idx]
            mu = 32 * tau * np.log(np.exp(sel / tau).mean(1))
            q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
            RAW.append(raw); NRM.append(xn.astype(np.float32))
            TCK.append(g.astype(np.uint8)); MU.append(mu.astype(np.float32))
            Q.append(q.astype(np.float32))
            obs, _, _, _ = env.step(torch.as_tensor(q, dtype=torch.float32, device=dev))
        print(f"  seed {sd}: {len(RAW)*a.envs:,} pairs so far  ({time.time()-t0:.0f}s)")

    RAW = np.concatenate(RAW); NRM = np.concatenate(NRM)
    TCK = np.concatenate(TCK); MU = np.concatenate(MU); Q = np.concatenate(Q)
    n = RAW.shape[0]
    print(f"\ntotal pairs {n:,}  obs dim {RAW.shape[1]}  act dim {Q.shape[1]}")

    lv = np.rint((Q + CL) / ST).astype(int)                  # 0..21 level index
    hist = np.stack([np.bincount(lv[:, o], minlength=LV) for o in range(6)])
    at_lo = (lv == 0).mean(0)
    at_hi = (lv == LV - 1).mean(0)
    print(f"fraction at -1 rail per dim : {np.round(at_lo*100, 2)}")
    print(f"fraction at +1 rail per dim : {np.round(at_hi*100, 2)}")
    print(f"fraction at EITHER rail     : {np.round((at_lo+at_hi)*100, 2)}  "
          f"overall {float(((lv == 0) | (lv == LV-1)).mean())*100:.2f}%")
    print(f"distinct levels used per dim: {[int((hist[o] > 0).sum()) for o in range(6)]}")
    print(f"pre-clip |mu|>1 fraction    : {float((np.abs(MU) > CL).mean())*100:.2f}%")

    meta = dict(
        note="SOFTWARE-TEACHER-VISITED states (OPEN-LOOP distribution). The spiking "
             "policy's own closed-loop distribution is NOT sampled here.",
        policy=os.path.basename(NPZ), n_pairs=int(n), obs_dim=int(RAW.shape[1]),
        act_dim=int(Q.shape[1]), envs=a.envs, steps=a.steps, seeds=seeds,
        n_rollouts=len(seeds) * a.envs, physics="solver 100 / ls 50, obs_clip_vel 10",
        out_levels=LV, out_clip=CL, out_step=ST,
        level_hist=hist.tolist(),
        frac_at_minus1=at_lo.tolist(), frac_at_plus1=at_hi.tolist(),
        frac_preclip_outside=float((np.abs(MU) > CL).mean()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, obs_raw=RAW, obs_norm=NRM, obs_bucket=TCK,
                        mu=MU, action=Q, meta=json.dumps(meta))
    print(f"\nwrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
