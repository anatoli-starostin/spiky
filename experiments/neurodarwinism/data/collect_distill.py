"""Collect a 100K-pair distillation dataset from the trained exp19 LUT actor.

WHAT IS THE TARGET. exp19's actor is `FastMultiHeadLut(..., exp_outputs=True,
exp_outputs_scale="sum")`. Its readout lives in
`src/spiky/lutorch/fast_multi_head_lut.py::_exp_outputs_fwd` (lines 165-188):

    d        = x[:, anchor_a] - x[:, anchor_b]                       # [B, T, NAP]
    index    = ((d > 0) * powers).sum(-1)                            # [B, T]
    w_sel    = weights.view(T*table_dim, n_out)[index + T_offset]    # [B, 1, T, 6]
    z        = clamp(w_sel / tau, -60, +60)
    lse      = logsumexp(z, dim=2)                                   # [B, 1, 6]
    out      = tph * tau * (lse - log(tph))                          # scale == "sum"

The PRIMARY target `y_prelog` is the quantity just inside that outer log:

    S[b, o] = sum_t exp( clamp(w_sel[b, 0, t, o] / tau, -60, 60) )   # == exp(lse)

i.e. the per-output sum of exponentials over the tph = 32 tables of the single head,
BEFORE the log and before the tph*tau scaling. It is strictly positive by construction.
The full action mean is recovered exactly as

    y_action_mean = tph * tau * ( log(S) - log(tph) )
                  = tph * tau * log( S / tph )

The clamp is inside the definition, faithfully: the dataset records what the network
actually computes, not an idealised version of it. (In practice it never binds here --
max |z| is reported by the sanity check.)

Because n_heads == 1 for the actor, "sum over the tables within each head" and "sum over
all 32 tables" are the same reduction; there is no head ambiguity.

HOW THE INPUTS ARE COLLECTED. On-policy rollouts of the trained actor in the same
GPU-batched MuJoCo-Warp env used for training (`warp_env.WarpWalker2dVecEnv`, identical solver
settings, NO velocity clipping -- exp19 was trained without it; the deploy .npz is the
separate velocity-clipped variant and is deliberately not used here).

The sim is driven by the DETERMINISTIC action mean plus a small Gaussian dither. The
dither is not decoration: reset noise is only 5e-3, so a purely deterministic policy
drives every parallel env down a near-identical trajectory and the "100K samples" would
collapse to a few hundred distinct states. The dither is a fraction of the policy's own
trained exploration std, so the visited states stay well inside the training
distribution. Targets are always recorded for the DETERMINISTIC readout at the visited
state -- the dither perturbs WHICH states are visited, never the label.

Usage:
  python collect_distill.py [--ckpt ../rerun_ckpt/actor_s1.pt] [--envs 250] [--steps 400]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
sys.path.insert(0, SRC)


def prelog_sum(lut, xn):
    """S = sum_t exp(clamp(w_sel/tau, +-clamp)) -- the pre-final-log quantity, [B, n_out].

    Mirrors `_exp_outputs_fwd` line for line; only the final `logsumexp` is replaced by an
    explicit exp().sum() so the summed exponentials themselves are observable.
    """
    n_tables = lut.soft_anchor_a_long.shape[0]
    n_out = lut.weights.shape[2]
    tau = lut.exp_outputs_tau
    with torch.no_grad():
        d = xn[:, lut.soft_anchor_a_long] - xn[:, lut.soft_anchor_b_long]
        bits = (d > 0).to(torch.int64)
        index = (bits * lut.soft_powers.view(1, 1, -1)).sum(dim=-1)
        offset = torch.arange(n_tables, device=lut.weights.device,
                              dtype=index.dtype) * lut.table_dim
        flat_idx = (index + offset.view(1, -1)).reshape(-1)
        w_sel = lut.weights.view(n_tables * lut.table_dim, n_out)[flat_idx]
        w_sel = w_sel.view(xn.shape[0], lut.n_heads, lut.tables_per_head, n_out)
        z = torch.clamp(w_sel / tau, min=-lut.exp_outputs_clamp, max=lut.exp_outputs_clamp)
        S = torch.exp(z.double()).sum(dim=2)                      # [B, 1, n_out], float64
        return S.squeeze(1), float(z.abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "actor_s1.pt"))
    ap.add_argument("--envs", type=int, default=250)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dither", type=float, default=0.3,
                    help="exploration noise as a FRACTION of the policy's trained std")
    ap.add_argument("--warmup", type=int, default=0,
                    help="steps to discard before recording (0 = record from reset)")
    ap.add_argument("--out", default=os.path.join(HERE, "distill_exp19_100k.npz"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from models import REGISTRY
    from warp_env import WarpWalker2dVecEnv

    dev = "cuda"
    torch.manual_seed(a.seed)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    print(f"checkpoint      : {os.path.relpath(a.ckpt, HERE)}")
    print(f"  arch          : {ck['arch']}  tph={ck['tables_per_head']}  seed={ck['seed']}")
    print(f"  final_ep_ret  : {ck['final_ep_ret']:.1f}   (training, stochastic policy)")
    # --obs-clip-vel/--solver-iters/--ls-iters postdate this checkpoint, so the keys are
    # absent from its config; the defaults below ARE the settings it was trained with.
    cfg = ck["config"]
    env_kw = dict(obs_clip_vel=cfg.get("obs_clip_vel"),
                  solver_iters=cfg.get("solver_iters") or 10,
                  ls_iters=cfg.get("ls_iters") or 8)
    print(f"  env           : obs_clip_vel={env_kw['obs_clip_vel']} "
          f"solver_iters={env_kw['solver_iters']} ls_iters={env_kw['ls_iters']}"
          + ("   (keys absent from the checkpoint config -> training defaults)"
             if "solver_iters" not in cfg else ""))

    ac = REGISTRY[ck["arch"]](ck["obs_dim"], ck["act_dim"],
                              tables_per_head=ck["tables_per_head"]).to(dev)
    ac.load_state_dict(ck["state_dict"])
    ac.eval()
    lut = ac.actor_lut
    tau = float(lut.exp_outputs_tau.detach())
    tph = int(lut.tables_per_head)
    std = float(ac.log_std.detach().exp().mean())
    print(f"  tau_actor     : {tau:.8f}   tph={tph}  n_heads={lut.n_heads}")
    print(f"  policy std    : {std:.4f}  -> dither std {a.dither * std:.4f}")

    obs_mean = ck["obs_mean"].to(dev)
    obs_var = ck["obs_var"].to(dev)

    env = WarpWalker2dVecEnv(num_envs=a.envs, device=dev, seed=a.seed, **env_kw)
    env.build_physics_graph()

    n_target = a.envs * a.steps
    X = np.empty((n_target, env.obs_dim), np.float32)
    XN = np.empty((n_target, env.obs_dim), np.float32)
    YP = np.empty((n_target, env.act_dim), np.float64)
    YA = np.empty((n_target, env.act_dim), np.float64)

    obs = env.reset()
    ep_ret = torch.zeros(a.envs, device=dev)
    ep_len = torch.zeros(a.envs, device=dev)
    fin_ret, fin_len = [], []
    zmax = 0.0
    rew_sum = healthy_sum = 0.0
    rew_steps = 0
    t0 = time.time()
    w = 0
    total_steps = a.warmup + a.steps
    for t in range(total_steps):
        xn = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
        with torch.no_grad():
            mean = lut(xn).squeeze(1)                      # the module's own readout
        S, zm = prelog_sum(lut, xn)
        zmax = max(zmax, zm)
        if t >= a.warmup:
            s = w * a.envs
            X[s:s + a.envs] = obs.cpu().numpy()
            XN[s:s + a.envs] = xn.cpu().numpy()
            YP[s:s + a.envs] = S.cpu().numpy()
            YA[s:s + a.envs] = mean.double().cpu().numpy()
            w += 1
        act = mean if a.dither == 0 else mean + a.dither * std * torch.randn_like(mean)
        obs, rew, term, trunc = env.step(act)
        done = (term | trunc)
        if t >= a.warmup:
            rew_sum += float(rew.sum())
            healthy_sum += float((~term).float().sum())
            rew_steps += a.envs
        ep_ret += rew
        ep_len += 1
        if done.any():
            fin_ret.append(ep_ret[done].cpu().numpy())
            fin_len.append(ep_len[done].cpu().numpy())
            ep_ret = ep_ret * (~done).float()
            ep_len = ep_len * (~done).float()
    el = time.time() - t0
    print(f"\nrolled {total_steps} steps x {a.envs} envs in {el:.1f}s "
          f"({n_target:,} recorded pairs)")

    fin_ret = np.concatenate(fin_ret) if fin_ret else np.array([])
    fin_len = np.concatenate(fin_len) if fin_len else np.array([])
    unfin_ret = ep_ret.cpu().numpy()
    unfin_len = ep_len.cpu().numpy()

    stats = {}
    print("\n--- rollout health (was the policy actually WALKING?) ---")
    if fin_ret.size:
        print(f"finished episodes : {fin_ret.size}   return mean {fin_ret.mean():.1f} "
              f"(min {fin_ret.min():.1f}, max {fin_ret.max():.1f})   "
              f"length mean {fin_len.mean():.1f}")
        stats["finished_episodes"] = int(fin_ret.size)
        stats["finished_return_mean"] = float(fin_ret.mean())
        stats["finished_return_min"] = float(fin_ret.min())
        stats["finished_return_max"] = float(fin_ret.max())
        stats["finished_length_mean"] = float(fin_len.mean())
    else:
        print("finished episodes : 0  (no env terminated or hit the time limit)")
        stats["finished_episodes"] = 0
    print(f"in-flight at end  : {unfin_ret.size} envs, return mean {unfin_ret.mean():.1f}, "
          f"length mean {unfin_len.mean():.1f}")
    ret_rate = rew_sum / max(rew_steps, 1)
    print(f"reward per step   : {ret_rate:.3f}  over all {rew_steps:,} recorded steps "
          f"(exp19's training arm averages ~5.4/step; an env that has fallen scores ~0 "
          f"and is reset)")
    print(f"healthy fraction  : {healthy_sum / max(rew_steps, 1):.4f}  "
          f"(fraction of recorded steps that did NOT terminate as unhealthy)")
    stats["healthy_fraction"] = healthy_sum / max(rew_steps, 1)
    stats["inflight_return_mean"] = float(unfin_ret.mean())
    stats["inflight_length_mean"] = float(unfin_len.mean())
    stats["reward_per_step"] = ret_rate

    print("\n--- array sanity ---")
    for nm, A in (("x", X), ("x_norm", XN), ("y_prelog", YP), ("y_action_mean", YA)):
        bad = int(np.isnan(A).sum() + np.isinf(A).sum())
        print(f"{nm:<14} {str(A.shape):<14} {str(A.dtype):<9} "
              f"min {A.min():>11.4f} max {A.max():>11.4f} "
              f"mean {A.mean():>9.4f} std {A.std():>8.4f}  nan/inf {bad}")
        stats[nm] = dict(shape=list(A.shape), dtype=str(A.dtype), nan_inf=bad,
                         min=float(A.min()), max=float(A.max()),
                         mean=float(A.mean()), std=float(A.std()))
    print(f"y_prelog strictly positive : {bool((YP > 0).all())}  (min {YP.min():.6f})")
    stats["y_prelog_strictly_positive"] = bool((YP > 0).all())
    print(f"max |w/tau| seen (clamp is {lut.exp_outputs_clamp:g}) : {zmax:.3f} "
          f"-> clamp {'BOUND' if zmax >= lut.exp_outputs_clamp else 'never bound'}")
    stats["max_abs_z"] = zmax
    stats["clamp"] = float(lut.exp_outputs_clamp)

    # distinctness: the LUT is a step function of the sign pattern, so what matters for a
    # student is how many DISTINCT inputs and how many distinct table-address patterns.
    uniq_x = len(np.unique(np.ascontiguousarray(XN).view(
        np.dtype((np.void, XN.dtype.itemsize * XN.shape[1])))))
    uniq_y = len(np.unique(np.ascontiguousarray(YA).view(
        np.dtype((np.void, YA.dtype.itemsize * YA.shape[1])))))
    print(f"distinct observations       : {uniq_x:,} / {n_target:,}")
    print(f"distinct output vectors     : {uniq_y:,} / {n_target:,}  "
          f"(the LUT has at most 2^6 rows x 32 tables of reachable patterns)")
    stats["distinct_observations"] = uniq_x
    stats["distinct_outputs"] = uniq_y

    print("\n--- pre-log reproduction check ---")
    recon = tph * tau * (np.log(YP) - math.log(tph))
    err = np.abs(recon - YA)
    print(f"|tph*tau*(log(y_prelog) - log(tph)) - y_action_mean|  "
          f"max {err.max():.3e}  mean {err.mean():.3e}")
    print(f"  (y_action_mean is the module's own logsumexp output; the residual is pure "
          f"fp32-vs-fp64 rounding)")
    stats["reproduction_err_max"] = float(err.max())
    stats["reproduction_err_mean"] = float(err.mean())
    rel = err.max() / max(float(np.abs(YA).max()), 1e-12)
    stats["reproduction_err_rel_max"] = float(rel)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        x=X, x_norm=XN,
        y_prelog=YP.astype(np.float32), y_action_mean=YA.astype(np.float32),
        y_prelog_f64=YP, y_action_mean_f64=YA,
        tau=np.float64(tau), tables_per_head=np.int64(tph),
        exp_clamp=np.float64(lut.exp_outputs_clamp),
        obs_mean=ck["obs_mean"].numpy(), obs_var=ck["obs_var"].numpy(),
        obs_count=np.float64(ck["obs_count"]),
        anchor_a=lut.soft_anchor_a_long.cpu().numpy(),
        anchor_b=lut.soft_anchor_b_long.cpu().numpy(),
        weights=lut.weights.detach().cpu().numpy(),
    )
    print(f"\nsaved -> {a.out}  ({os.path.getsize(a.out) / 1e6:.2f} MB)")

    meta = dict(
        source_checkpoint=os.path.relpath(os.path.abspath(a.ckpt),
                                          os.path.abspath(os.path.join(HERE, "..", ".."))),
        arch=ck["arch"], tables_per_head=tph, seed_of_policy=ck["seed"],
        policy_final_ep_ret=ck["final_ep_ret"],
        tau=tau, exp_clamp=float(lut.exp_outputs_clamp),
        n_envs=a.envs, n_steps=a.steps, warmup=a.warmup, n_pairs=n_target,
        dither_fraction=a.dither, policy_std=std, dither_std=a.dither * std,
        env=dict(**env_kw, reset_noise=5e-3, max_steps=1000, seed=a.seed),
        collection_seed=a.seed, wall_s=round(el, 1), stats=stats,
    )
    mp = os.path.join(os.path.dirname(a.out), "meta.json")
    json.dump(meta, open(mp, "w"), indent=1)
    print(f"saved -> {mp}")


if __name__ == "__main__":
    main()
