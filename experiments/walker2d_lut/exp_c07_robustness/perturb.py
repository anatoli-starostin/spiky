"""exp_c07 — perturbed Walker2d dynamics + a batched CPU evaluator (#75).

Zero-shot robustness: every policy is FROZEN (weights and, for the LUT policies, the
stored observation standardiser). Nothing is re-fitted to a perturbed environment —
re-fitting the obs mean/std would leak knowledge of the new dynamics and stop this
being zero-shot.

Framework-agnostic on purpose: a policy is just `f(obs[N,17]) -> act[N,6]`, so the
torch and JAX policies (which live in separate venvs) run through the identical
environment code and are therefore comparable.

Speed note: the obvious implementation — one episode at a time, one policy call per
step — costs ~8 hours for the full grid, because a single-sample forward is dominated
by framework overhead (~1 ms) while a MuJoCo step is 0.067 ms. Stepping all `episodes`
environments in lockstep and issuing ONE batched policy call per tick makes the policy
overhead amortise; the same grid then takes minutes.
"""
import numpy as np
import mujoco

XML = ("/home/astarostin/projects/spiky/.venv/lib/python3.12/site-packages/"
       "gymnasium/envs/mujoco/assets/walker2d_v5.xml")
FRAME_SKIP = 4

AXES = {
    "mass":     [0.7, 0.85, 1.0, 1.15, 1.3],
    "gravity":  [0.85, 1.0, 1.15],
    "friction": [0.5, 0.75, 1.0, 1.5, 2.0],
    "geometry": [0.9, 0.95, 1.0, 1.05, 1.1],
}


def make_model(axis=None, value=1.0, xml=XML):
    """Stock Walker2d-v5 model with ONE axis perturbed. axis=None -> nominal."""
    m = mujoco.MjModel.from_xml_path(xml)
    if axis is None or value == 1.0:
        return m
    if axis == "mass":
        m.body_mass[:] *= value
        m.body_inertia[:] *= value          # keep inertia consistent with mass
    elif axis == "gravity":
        m.opt.gravity[2] *= value
    elif axis == "friction":
        m.geom_friction[:, 0] *= value      # sliding friction
    elif axis == "geometry":
        # Scale the ROBOT only, not the floor (geom 0 is the ground plane).
        m.geom_size[1:] *= value
        m.body_pos[1:] *= value             # segment lengths / joint offsets
    else:
        raise ValueError(axis)
    return m


def eval_batched(model, policy_fn, episodes=100, max_steps=1000, seed0=0):
    """Deterministic eval of `episodes` episodes stepped in lockstep.

    Reward/termination reproduce Walker2d-v5 exactly:
      r = healthy(1.0) + dx/dt - 1e-3*||a||^2 ; unhealthy if z not in (0.8,2.0)
      or |angle| > 1 ; truncate at 1000 steps ; reset noise U(-5e-3, 5e-3).
    """
    dt = model.opt.timestep * FRAME_SKIP
    datas, alive, rets = [], np.ones(episodes, bool), np.zeros(episodes)
    for ep in range(episodes):
        d = mujoco.MjData(model)
        rng = np.random.default_rng(seed0 + ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, model.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, model.nv)
        mujoco.mj_forward(model, d)
        datas.append(d)

    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        obs = np.stack([np.concatenate([datas[i].qpos[1:],
                                        np.clip(datas[i].qvel, -10, 10)])
                        for i in idx]).astype(np.float32)
        act = np.clip(np.asarray(policy_fn(obs), np.float64), -1.0, 1.0)
        for j, i in enumerate(idx):
            d = datas[i]
            x0 = d.qpos[0]
            d.ctrl[:] = act[j]
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, d)
            rets[i] += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(act[j] @ act[j])
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                alive[i] = False
    return float(rets.mean()), float(rets.std()), rets


def sweep(policy_fn, name, episodes=100, log=print):
    """Full grid for one policy -> list of row dicts."""
    rows = []
    nominal = None
    for axis, values in AXES.items():
        for v in values:
            m = make_model(axis, v)
            mean, sd, _ = eval_batched(m, policy_fn, episodes=episodes)
            if v == 1.0 and nominal is None:
                nominal = mean
            rows.append(dict(policy=name, axis=axis, value=v, mean=mean, std=sd,
                             episodes=episodes, solved=bool(mean >= 3000)))
            log(f"  {name:<16} {axis:<9} x{v:<5} -> {mean:8.1f} +/- {sd:6.1f}"
                f"  {'' if mean >= 3000 else '(below 3000)'}")
    return rows
