"""exp_c02d — comparability cross-check: does the 10/8 solver change matter? (#75)

The question that decides whether the MJX track is usable: a policy trained under
MJX's reduced solver (10/8) must behave sensibly in the CPU reference env
(`Walker2d-v5`, solver 100/50), because the reference is what the SAC baseline and
the LUT results are measured in.

Two independent measurements, both on the SAME policy parameters:

  A. Open-loop dynamics divergence. Drive identical action sequences from identical
     initial states through MJX@10/8, MJX@100/50 and CPU MuJoCo, and measure state
     drift. This isolates the solver from the policy.
  B. Closed-loop return transfer. Evaluate the trained policy in
       - MJX @ 10/8   (what it trained in)
       - MJX @ 100/50 (stock dynamics, same engine)
       - CPU Walker2d-v5 (the reference)
     A small gap 10/8 -> reference means MJX-trained policies transfer.

Run AFTER ppo_mjx.py --save-params.

Usage:
    XLA_PYTHON_CLIENT_PREALLOCATE=false python cross_check.py --params ppo_policy.msgpack
"""
import argparse, json, os

import jax, jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

import mjx_walker2d as W
from ppo_mjx import ActorCritic, OBS

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# A. open-loop dynamics divergence
# --------------------------------------------------------------------------
def rollout_mjx(iters, ls, actions, qpos0, qvel0):
    """actions [T, 6] -> trajectory of qpos [T, nq] under MJX at a solver setting."""
    m = W.make_model(iters, ls)
    mx = mjx.put_model(m)
    d = mjx.make_data(mx).replace(qpos=jnp.array(qpos0), qvel=jnp.array(qvel0))
    d = mjx.forward(mx, d)

    @jax.jit
    def step_n(d, a):
        def phys(d, _):
            return mjx.step(mx, d.replace(ctrl=a)), None
        d, _ = jax.lax.scan(phys, d, None, length=W.FRAME_SKIP)
        return d, d.qpos
    traj = []
    for a in actions:
        d, q = step_n(d, jnp.array(a))
        traj.append(np.asarray(q))
    return np.stack(traj)


def rollout_cpu(actions, qpos0, qvel0):
    m = mujoco.MjModel.from_xml_path(W.XML)      # stock 100/50 from the XML
    d = mujoco.MjData(m)
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    mujoco.mj_forward(m, d)
    traj = []
    for a in actions:
        d.ctrl[:] = a
        for _ in range(W.FRAME_SKIP):
            mujoco.mj_step(m, d)
        traj.append(d.qpos.copy())
    return np.stack(traj)


def open_loop(T=200, seed=0):
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_path(W.XML)
    d0 = mujoco.MjData(m)
    qpos0 = d0.qpos.copy() + rng.uniform(-5e-3, 5e-3, m.nq)
    qvel0 = d0.qvel.copy() + rng.uniform(-5e-3, 5e-3, m.nv)
    # smooth action sequence (a constant torque excites contacts unrealistically)
    t = np.arange(T)[:, None]
    actions = 0.6 * np.sin(t / 12.0 + rng.uniform(0, 6.28, (1, 6))).astype(np.float64)

    cpu = rollout_cpu(actions, qpos0, qvel0)
    a108 = rollout_mjx(10, 8, actions, qpos0, qvel0)
    a100 = rollout_mjx(100, 50, actions, qpos0, qvel0)

    def div(a, b):
        e = np.linalg.norm(a - b, axis=-1)
        return dict(mean=float(e.mean()), at_50=float(e[49]), at_200=float(e[-1]),
                    max=float(e.max()))
    return dict(steps=T,
                mjx100_vs_cpu=div(a100, cpu),
                mjx10_8_vs_cpu=div(a108, cpu),
                mjx10_8_vs_mjx100=div(a108, a100))


# --------------------------------------------------------------------------
# B. closed-loop return transfer
# --------------------------------------------------------------------------
def load_policy(path):
    import flax.serialization as fs
    net = ActorCritic()
    shape = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS)))
    with open(path, "rb") as f:
        params = fs.from_bytes(shape, f.read())
    return net, params


def eval_mjx(net, params, iters, ls, n_envs=256, max_steps=1000, seed=0):
    """Deterministic (mean-action) return over n_envs parallel episodes."""
    m = W.make_model(iters, ls)
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(jax.random.PRNGKey(seed), n_envs))

    @jax.jit
    def run(st):
        def one(carry, _):
            st, ret, alive = carry
            mean, _, _ = net.apply(params, st.obs)
            nst = v_step(st, mean)
            ret = ret + nst.reward * alive
            alive = alive * (1.0 - nst.done)
            return (nst, ret, alive), None
        (st, ret, alive), _ = jax.lax.scan(
            one, (st, jnp.zeros(n_envs), jnp.ones(n_envs)), None, length=max_steps)
        return ret
    r = np.asarray(run(st))
    return float(r.mean()), float(r.std())


def eval_cpu(net, params, episodes=20, seed=0):
    """The reference number: gymnasium Walker2d-v5, CPU MuJoCo, stock solver.

    gymnasium lives in the OTHER venv, so this runs the env directly on mujoco with
    Walker2d-v5's documented obs/reward/termination (identical to mjx_walker2d.py).
    """
    m = mujoco.MjModel.from_xml_path(W.XML)
    dt = m.opt.timestep * W.FRAME_SKIP
    rng = np.random.default_rng(seed)
    rets = []
    for _ in range(episodes):
        d = mujoco.MjData(m)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, m.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, m.nv)
        mujoco.mj_forward(m, d)
        R = 0.0
        for _ in range(1000):
            obs = np.concatenate([d.qpos[1:], np.clip(d.qvel, -10, 10)])
            mean, _, _ = net.apply(params, jnp.array(obs, jnp.float32)[None])
            a = np.clip(np.asarray(mean)[0], -1, 1)
            x0 = d.qpos[0]
            d.ctrl[:] = a
            for _ in range(W.FRAME_SKIP):
                mujoco.mj_step(m, d)
            z, ang = d.qpos[1], d.qpos[2]
            healthy = (0.8 < z < 2.0) and (-1.0 < ang < 1.0)
            R += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(np.sum(a ** 2))
            if not healthy:
                break
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="ppo_policy.msgpack")
    ap.add_argument("--cpu-episodes", type=int, default=20)
    ap.add_argument("--mjx-envs", type=int, default=256)
    a = ap.parse_args()

    print("=== A. open-loop dynamics divergence (identical actions & init state) ===")
    ol = open_loop()
    for k, v in ol.items():
        if k == "steps":
            continue
        print(f"  {k:22} mean |dqpos| {v['mean']:.4f} | @50 {v['at_50']:.4f} "
              f"| @200 {v['at_200']:.4f} | max {v['max']:.4f}")

    out = dict(open_loop=ol)
    path = os.path.join(HERE, a.params)
    if os.path.exists(path):
        print("\n=== B. closed-loop return transfer (same policy, three engines) ===")
        net, params = load_policy(path)
        m108 = eval_mjx(net, params, 10, 8, n_envs=a.mjx_envs)
        m100 = eval_mjx(net, params, 100, 50, n_envs=a.mjx_envs)
        cpu = eval_cpu(net, params, episodes=a.cpu_episodes)
        print(f"  MJX @ 10/8   (trained in)  {m108[0]:8.1f} +/- {m108[1]:6.1f}")
        print(f"  MJX @ 100/50 (stock)       {m100[0]:8.1f} +/- {m100[1]:6.1f}")
        print(f"  CPU Walker2d-v5 (REF)      {cpu[0]:8.1f} +/- {cpu[1]:6.1f}")
        rel = 100.0 * (cpu[0] - m108[0]) / max(abs(m108[0]), 1e-9)
        print(f"  -> transfer gap 10/8 -> reference: {rel:+.1f}%")
        out["closed_loop"] = dict(mjx_10_8=m108, mjx_100_50=m100, cpu_reference=cpu,
                                  transfer_gap_pct=round(rel, 1))
    else:
        print(f"\n(no policy at {path}; run ppo_mjx.py --save-params first)")

    with open(os.path.join(HERE, "cross_check_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote cross_check_results.json")


if __name__ == "__main__":
    main()
