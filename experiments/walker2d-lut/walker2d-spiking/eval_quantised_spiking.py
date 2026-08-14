"""Two validation evals for the amplitude-encoded quantised spiking actor.

(b) SURROGATE, full tier: the software quantised policy in the fast batched harness, with
    the SNN's MEASURED per-dim off-by-one-level error injected. Paired against the same
    harness with no jitter, so the delta is clean.
(a) REAL, reduced tier: the actual SNN in the loop of the Warp Walker2d env.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/walker2d-lut/src")
from warp_env import WarpWalker2dVecEnv                       # noqa: E402
import tiny_lut_quantised_pipeline as QP                      # noqa: E402

NPZ = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/"
       "exp19_lut-lse-expmlpcrit-t32/deploy/quantised/"
       "walker2d_fastlut_lse_exp19_quantised.npz")


def software_actions(xn, Z, tau, CL, ST):
    """the shipped quantised policy, batched, on already-normalised obs (torch)"""
    g = torch.searchsorted(Z["edges"], xn.reshape(-1).contiguous()).reshape(xn.shape)
    xq = Z["dequant"][g.clamp(0, Z["dequant"].numel() - 1)]
    d = xq[:, Z["aa"]] - xq[:, Z["bb"]]
    idx = ((d > 0).to(torch.int64) * Z["pw"]).sum(-1)
    sel = Z["W"][torch.arange(32, device=xn.device)[None, :], idx]
    mu = 32 * tau * torch.log(torch.exp(sel / tau).mean(1))
    q = ((mu.clamp(-CL, CL) + CL) / ST).round() * ST - CL
    return q.clamp(-CL, CL), mu


def rollout_surrogate(env, Z, tau, CL, ST, steps, jitter, dev, seed=0):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    obs = env.reset()
    ep = torch.zeros(env.N, device=dev)
    done_rets = []
    with torch.no_grad():
        for _ in range(steps):
            xn = (obs - Z["om"]) / torch.sqrt(Z["ov"] + 1e-8)
            a, _ = software_actions(xn, Z, tau, CL, ST)
            if jitter is not None:
                r = torch.rand(a.shape, device=dev, generator=g)
                hit = r < jitter[None, :]
                sgn = torch.where(torch.rand(a.shape, device=dev, generator=g) < 0.5,
                                  -1.0, 1.0)
                a = (a + hit.float() * sgn * ST).clamp(-CL, CL)
            obs, rew, term, trunc = env.step(a)
            ep += rew
            d_ = term | trunc
            if d_.any():
                done_rets.append(ep[d_].clone()); ep = ep * (~d_).float()
    return torch.cat(done_rets).cpu().numpy() if done_rets else np.zeros(0)


def rollout_real(env, net, ids, n_ticks, edges, om, ov, aff, CL, ST, steps, dev):
    """the actual SNN in the loop, batched over envs"""
    B = env.N
    obs = env.reset()
    ep = torch.zeros(B, device=dev)
    done_rets, lens, ep_len = [], [], torch.zeros(B, device=dev)
    for _ in range(steps):
        xn = ((obs - om) / torch.sqrt(ov + 1e-8)).cpu().numpy()
        ticks = QP.encode_gauss(xn, edges)
        o = QP.run(net, ids, ticks, n_ticks, dev)
        T = o[6].astype(np.float64)
        # self-timed: reference the crossing to the completion event (t_last)
        tl = ticks.max(1)[:, None].astype(np.float64)
        mu = aff[:, 0][None, :] * (T - tl) + aff[:, 1][None, :]
        mu = np.where(T >= n_ticks, -CL, mu)
        q = np.clip(np.round((np.clip(mu, -CL, CL) + CL) / ST) * ST - CL, -CL, CL)
        obs, rew, term, trunc = env.step(torch.as_tensor(q, dtype=torch.float32, device=dev))
        ep += rew; ep_len += 1
        d_ = term | trunc
        if d_.any():
            done_rets.append(ep[d_].clone()); lens.append(ep_len[d_].clone())
            ep = ep * (~d_).float(); ep_len = ep_len * (~d_).float()
    r = torch.cat(done_rets).cpu().numpy() if done_rets else np.zeros(0)
    ln = torch.cat(lens).cpu().numpy() if lens else np.zeros(0)
    return r, ln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sur-envs", type=int, default=1024)
    ap.add_argument("--sur-steps", type=int, default=2000)
    ap.add_argument("--real-envs", type=int, default=64)
    ap.add_argument("--real-steps", type=int, default=1200)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")

    Zn = np.load(NPZ)
    C = json.load(open(a.calib))
    tau = float(Zn["tau_actor"])
    CL, LV = float(Zn["out_quant_clip"]), int(Zn["out_quant_levels"])
    ST = 2 * CL / (LV - 1)
    exact = np.array([C["stage3"][str(o)]["exact"] for o in range(6)])
    jit = 1.0 - exact
    print(f"measured per-dim off-by-one rates: {np.round(jit*100, 2)}  "
          f"(mean {jit.mean()*100:.2f}%)")

    Z = dict(edges=torch.tensor(Zn["in_quant_edges"], dtype=torch.float32, device=dev),
             dequant=torch.tensor(Zn["in_quant_dequant"], dtype=torch.float32, device=dev),
             aa=torch.tensor(Zn["anchor_a"], device=dev),
             bb=torch.tensor(Zn["anchor_b"], device=dev),
             W=torch.tensor(Zn["weights"], dtype=torch.float32, device=dev),
             om=torch.tensor(Zn["obs_mean"], dtype=torch.float32, device=dev),
             ov=torch.tensor(Zn["obs_var"], dtype=torch.float32, device=dev),
             pw=torch.tensor(1 << np.arange(5, -1, -1), device=dev))
    res = {}

    # ---------------- (b) surrogate, full tier ----------------
    env = WarpWalker2dVecEnv(num_envs=a.sur_envs, seed=0, solver_iters=100, ls_iters=50,
                             obs_clip_vel=10.0)
    for lbl, j in (("baseline_no_jitter", None), ("jittered", torch.tensor(
            jit, dtype=torch.float32, device=dev))):
        t0 = time.time()
        r = rollout_surrogate(env, Z, tau, CL, ST, a.sur_steps, j, dev)
        res[lbl] = dict(n=int(len(r)), mean=float(r.mean()), sd=float(r.std()),
                        median=float(np.median(r)),
                        se=float(r.std() / np.sqrt(len(r))), wall_s=round(time.time() - t0, 1))
        print(f"(b) {lbl:20s} n={len(r):5d} mean {r.mean():7.1f} +- {r.std():6.1f}  "
              f"se {res[lbl]['se']:.1f}  median {np.median(r):7.1f}")
    d = res["jittered"]["mean"] - res["baseline_no_jitter"]["mean"]
    se = np.hypot(res["jittered"]["se"], res["baseline_no_jitter"]["se"])
    res["delta"] = dict(value=float(d), se=float(se), sigma=float(d / se))
    print(f"(b) DELTA {d:+.1f} +- {se:.1f}  ({d/se:+.2f} sigma)")

    # ---------------- (a) real SNN in the loop ----------------
    if not a.skip_real:
        Zq = np.load("/home/astarostin/projects/spiky/experiments/walker2d-lut/walker2d-spiking/"
                     "deploy_quantised/spiking_lut_quantised_actor.npz")
        net, ids, nsyn, n_ticks, nneur, aff_, win, beta, dmax = QP.build(
            Zn, list(range(6)), False, float(Zq["tau_m_out"]), "cuda", 6, 3.0,
            True)                       # tie_break=False, gt_skew=True
        aff = Zq["affine"].astype(np.float64)
        env2 = WarpWalker2dVecEnv(num_envs=a.real_envs, seed=0, solver_iters=100,
                                  ls_iters=50, obs_clip_vel=10.0)
        t0 = time.time()
        r, ln = rollout_real(env2, net, ids, n_ticks, Zn["in_quant_edges"].astype(np.float64),
                             torch.tensor(Zn["obs_mean"], dtype=torch.float32, device=dev),
                             torch.tensor(Zn["obs_var"], dtype=torch.float32, device=dev),
                             aff, CL, ST, a.real_steps, dev)
        res["real_snn"] = dict(n=int(len(r)), mean=float(r.mean()) if len(r) else None,
                               sd=float(r.std()) if len(r) else None,
                               se=float(r.std() / np.sqrt(len(r))) if len(r) else None,
                               mean_len=float(ln.mean()) if len(ln) else None,
                               n_ticks=int(n_ticks), wall_s=round(time.time() - t0, 1))
        print(f"(a) real SNN         n={len(r):5d} mean "
              f"{r.mean() if len(r) else float('nan'):7.1f} +- "
              f"{r.std() if len(r) else float('nan'):6.1f}  "
              f"se {res['real_snn']['se'] if len(r) else float('nan')}  "
              f"mean_len {ln.mean() if len(ln) else float('nan'):.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
