"""exp_c25 — deterministic 100-episode CPU eval of a PHASE-AUGMENTED LUT-SAC actor.

Same protocol as exp_c09's `eval_cpu.py` and `perturb.eval_batched`: gymnasium's own
Walker2d-v5 model on CPU MuJoCo at the stock solver, 100 episodes seeded 0..99, the
deterministic action tanh(mu). The ONLY difference is that the policy is handed a
19-dim observation whose last two channels are the same clock the trainer used.

WHY THE CLOCK CANNOT BE PASSED BY HAND. A phase-aware actor evaluated at the wrong
frequency is not a slightly worse policy, it is a different one -- so `--phase-freq`
is not an argument here. The frequency and control dt are read out of the checkpoint,
where the trainer wrote them, and a checkpoint without them is rejected rather than
guessed at.

Phase alignment with training: `perturb.eval_batched` steps every surviving episode in
lockstep, so at loop iteration `step` (1-based) every alive episode has taken exactly
step-1 control steps since its own reset -- there is no per-episode counter to keep.
phi = omega_dt * (step - 1), which is 0 on the first decision, matching the trainer's
`steps == 0` at episode start.

Writes <actor>_phase_cpueval.json (a distinct suffix, so it can never collide with the
exp_c09 `_cpueval.json` files).

Usage:
  python eval_phase_cpu.py <actor.npz> [--episodes 100]
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac"):
    sys.path.insert(0, os.path.join(D, p))

import jax                                                 # noqa: E402
import jax.numpy as jnp                                    # noqa: E402
import jax_lut_ext as X                                    # noqa: E402
import perturb                                             # noqa: E402
import phase_lut_sac as P                                  # noqa: E402  (same dir)

ACT = 6


def load_phase_actor(path, forward_mode="hard"):
    """-> (make_policy_fn, n_params, phase_freq, omega_dt).

    `make_policy_fn()` returns a FRESH closure with its own step counter, so a caller
    can evaluate twice without the second run inheriting the first one's phase.
    """
    z = np.load(path)
    for k in ("phase_freq", "dt_ctrl"):
        if k not in z.files:
            raise SystemExit(
                f"{os.path.basename(path)} has no '{k}'. This eval refuses to guess a "
                f"clock: an actor trained with phase and scored without it (or at the "
                f"wrong f) is a different policy, and the number would be meaningless. "
                f"Re-train with phase_lut_sac.py, which writes both fields.")
    f = float(z["phase_freq"])
    dt = float(z["dt_ctrl"])
    omega_dt = 2.0 * np.pi * f * dt

    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    mean, scale = P.ext_norm(jnp.asarray(stats["obs_mean"], jnp.float32),
                             jnp.asarray(stats["obs_std"], jnp.float32))
    p = {k: jnp.asarray(z[k]) for k in ("w", "b", "weights", "log_T_soft", "log_T_sel")}
    heads, tph = int(z["n_heads"]), int(z["tph"])
    if p["w"].shape[-1] != P.AOBS:
        raise SystemExit(f"addressing expects {p['w'].shape[-1]} inputs but this eval "
                         f"builds {P.AOBS}. Refusing to run a mismatched actor.")

    @jax.jit
    def act(aug):
        x = (aug - mean) / scale
        y = X.apply(forward_mode)(x, p["w"], p["b"], p["weights"],
                                  p["log_T_soft"], p["log_T_sel"],
                                  heads, tph).sum(1)
        return jnp.tanh(y[:, :ACT])

    n = int(sum(np.prod(z[k].shape) for k in ("w", "b", "weights")))

    def make():
        state = dict(step=0)

        def fn(obs):
            phi = omega_dt * state["step"]
            state["step"] += 1
            ph = np.broadcast_to(np.array([np.sin(phi), np.cos(phi)], np.float32),
                                 (obs.shape[0], 2))
            return np.asarray(act(jnp.asarray(np.concatenate([obs, ph], 1))))
        return fn

    return make, n, f, omega_dt


def evaluate(make_fn, episodes=100, max_steps=1000, seed0=0):
    """perturb.eval_batched verbatim + the alive mask and the forward speed."""
    m = perturb.make_model(None, 1.0)
    dt = m.opt.timestep * perturb.FRAME_SKIP
    fn = make_fn()
    datas, alive, rets = [], np.ones(episodes, bool), np.zeros(episodes)
    length, x0_all = np.zeros(episodes, int), np.zeros(episodes)
    for ep in range(episodes):
        d = mujoco.MjData(m)
        rng = np.random.default_rng(seed0 + ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, m.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, m.nv)
        mujoco.mj_forward(m, d)
        datas.append(d)
        x0_all[ep] = d.qpos[0]

    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        obs = np.stack([np.concatenate([datas[i].qpos[1:],
                                        np.clip(datas[i].qvel, -10, 10)])
                        for i in idx]).astype(np.float32)
        act_ = np.clip(np.asarray(fn(obs), np.float64), -1.0, 1.0)
        for j, i in enumerate(idx):
            d = datas[i]
            xb = d.qpos[0]
            d.ctrl[:] = act_[j]
            for _ in range(perturb.FRAME_SKIP):
                mujoco.mj_step(m, d)
            rets[i] += 1.0 + (d.qpos[0] - xb) / dt - 1e-3 * float(act_[j] @ act_[j])
            length[i] += 1
            z_, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z_ < 2.0 and -1.0 < ang < 1.0):
                alive[i] = False
    vel = np.array([(datas[i].qpos[0] - x0_all[i]) / max(length[i], 1) / dt
                    for i in range(episodes)])
    return (float(rets.mean()), float(rets.std(ddof=1)), int(alive.sum()),
            float(vel.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    path = a.actor if os.path.isabs(a.actor) else os.path.join(HERE, a.actor)
    make, n, f, _ = load_phase_actor(path)
    mean, sd, full, vel = evaluate(make, episodes=a.episodes)
    print(f"{os.path.basename(path)} ({n:,} params, phase {f:.4f} Hz) | "
          f"CPU-reference {a.episodes}-ep deterministic: {mean:.1f} +/- {sd:.1f}  "
          f"full {full}/{a.episodes}  vel {vel:.3f} m/s", flush=True)
    out = path.replace("_actor.npz", "_phase_cpueval.json")
    json.dump(dict(actor=os.path.basename(path), params=n, phase_freq=f,
                   episodes=a.episodes, forward_mode="hard",
                   cpu_reference_mean=mean, cpu_reference_std=sd,
                   full_length=full, vel_mean=vel), open(out, "w"), indent=1)
    print(f"wrote {os.path.basename(out)}", flush=True)


if __name__ == "__main__":
    main()
