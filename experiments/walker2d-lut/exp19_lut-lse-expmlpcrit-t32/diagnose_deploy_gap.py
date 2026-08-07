"""Why does the exp19 actor fall on the server when it walks locally?

Checks, cheapest first:

A. The committed artifact is intact and is the seed we think it is.
B. TRAINING vs DEPLOYMENT observation mismatch. Our training env (warp_env.py) builds
   obs = concat(qpos[1:], qvel) with NO velocity clipping. Gymnasium's Walker2d clips
   velocity to [-10, 10] before handing it over. If the walker exceeds that, the policy
   sees a different vector at deployment than it ever saw in training -- and the
   normalisation statistics were computed on the UNCLIPPED distribution.
C. The server harness reproduced exactly: reset(seed=0), continuous stepping with the
   server's auto-restart, actor constructed the way the server constructs it.
D. Does the very first action already look wrong (i.e. does it fall from step 1)?

Usage:  python diagnose_deploy_gap.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(HERE, "deploy")
NPZ = os.path.join(DEPLOY, "walker2d_fastlut_lse_exp19.npz")


def act_np(P, obs):
    x = (np.asarray(obs, np.float64).reshape(-1)[:17] - P["obs_mean"])
    x = x / np.sqrt(P["obs_var"] + 1e-8)
    d = x[P["anchor_a"]] - x[P["anchor_b"]]
    nap = P["anchor_a"].shape[1]
    addr = ((d > 0).astype(np.int64) * (1 << np.arange(nap - 1, -1, -1))).sum(-1)
    T = P["weights"].shape[0]
    sel = P["weights"][np.arange(T), addr]
    z = sel / float(P["tau_actor"])
    m = z.max(axis=0)
    lse = m + np.log(np.exp(z - m).sum(axis=0))
    return np.clip(T * float(P["tau_actor"]) * (lse - np.log(T)), -1, 1).astype(np.float32)


def main():
    import gymnasium as gym
    import mujoco
    print(f"gymnasium {gym.__version__}   mujoco {mujoco.__version__}   numpy {np.__version__}")
    print("(server image pins gymnasium 1.3.0, mujoco 3.11.0, numpy 2.5.1)\n")

    P = {k: np.load(NPZ)[k] for k in np.load(NPZ).files}
    print("A. artifact")
    print(f"   keys {sorted(P)}")
    print(f"   weights {P['weights'].shape}  tau {float(P['tau_actor']):.6f}")
    print(f"   obs_mean[:4] {np.round(P['obs_mean'][:4], 4)}")
    print(f"   obs_var [:4] {np.round(P['obs_var'][:4], 4)}")

    # ---- B. does deployment clip velocities that training never clipped? ----
    print("\nB. observation mismatch: gymnasium clips qvel to [-10,10]; warp_env does NOT")
    env = gym.make("Walker2d-v5")
    obs, _ = env.reset(seed=0)
    raw_hits = 0
    n = 0
    vmax = 0.0
    for ep in range(5):
        obs, _ = env.reset(seed=100 + ep)
        done = False
        while not done:
            a = act_np(P, obs)
            obs, r, term, trunc, _ = env.step(a)
            qvel = np.asarray(env.unwrapped.data.qvel).ravel()
            vmax = max(vmax, float(np.abs(qvel).max()))
            raw_hits += int((np.abs(qvel) > 10.0).sum())
            n += qvel.size
            done = term or trunc
    print(f"   |qvel| > 10 in {raw_hits}/{n} components ({100*raw_hits/max(n,1):.3f}%), "
          f"max |qvel| {vmax:.2f}")
    print("   -> the obs the policy sees at deployment IS clipped where that happens;")
    print("      training never clipped, and obs_var was measured on unclipped data.")

    # ---- C. the server harness, reproduced exactly -------------------------
    print("\nC. server harness (reset(seed=0), continuous stepping, auto-restart on done)")
    env2 = gym.make("Walker2d-v5")
    o, _ = env2.reset(seed=0)
    ep_ret, ep_len, rets, lens = 0.0, 0, [], []
    for step in range(3000):
        a = act_np(P, o)
        o, r, term, trunc, _ = env2.step(a)
        ep_ret += r
        ep_len += 1
        if term or trunc:
            rets.append(ep_ret)
            lens.append(ep_len)
            ep_ret, ep_len = 0.0, 0
            o, _ = env2.reset()
    print(f"   {len(rets)} episodes in 3000 steps: lengths {lens[:10]}")
    if rets:
        print(f"   returns {[round(v,1) for v in rets[:10]]}")
        print(f"   mean length {np.mean(lens):.1f}  mean return {np.mean(rets):.1f}")

    # ---- D. the opening actions -------------------------------------------
    print("\nD. first few actions from the reset state")
    o, _ = env2.reset(seed=0)
    for i in range(3):
        a = act_np(P, o)
        print(f"   step {i}: action {np.round(a, 3)}")
        o, r, term, trunc, _ = env2.step(a)
        print(f"           reward {r:+.3f}  terminated {term}  z {env2.unwrapped.data.qpos[1]:.3f}")


if __name__ == "__main__":
    main()
