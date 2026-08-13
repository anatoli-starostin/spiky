"""Walker2d-v5 reimplemented on MJX — batched, jittable, GPU-resident (issue #75).

Loads gymnasium's own `walker2d_v5.xml`, and reproduces Walker2d-v5's observation,
reward and termination *exactly as documented*:

  obs         qpos[1:] (8)  ++  clip(qvel, -10, 10) (9)          -> 17
  reward      healthy(1.0) + forward_weight * dx/dt - 1e-3*||a||^2
  terminate   z not in [0.8, 2.0]  or  |torso angle| > 1
  dt          timestep * frame_skip = 0.002 * 4 = 0.008
  reset       qpos/qvel + U(-5e-3, +5e-3)
  truncate    1000 steps

Deliberately dependency-light: no brax (which is unmaintained and broke against
jax 0.11), just mjx + jax. Everything is a pure function of (state, action) so it
vmaps over thousands of envs and jits into one kernel.
"""
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

GYM_ASSETS = os.path.expanduser(
    "~/projects/spiky/.venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets")
XML = os.environ.get("WALKER2D_XML", os.path.join(GYM_ASSETS, "walker2d_v5.xml"))

FRAME_SKIP = 4
HEALTHY_Z = (0.8, 2.0)
HEALTHY_ANGLE = (-1.0, 1.0)
HEALTHY_REWARD = 1.0
FORWARD_WEIGHT = 1.0
CTRL_COST = 1e-3
RESET_NOISE = 5e-3
MAX_STEPS = 1000


class State(NamedTuple):
    data: object          # mjx.Data (batched under vmap)
    obs: jnp.ndarray      # [17]
    reward: jnp.ndarray
    done: jnp.ndarray
    steps: jnp.ndarray
    key: jnp.ndarray


# Approved solver setting for this track (Anatoli, 2026-07-28): 10/8.
# The XML ships 100/50, which is what CPU MuJoCo (and therefore the Walker2d-v5 SAC
# reference) uses. Reducing iterations is documented MJX practice for GPU throughput
# and DOES change dynamics slightly — see cross_check.py for the measured effect.
SOLVER_ITERATIONS = 10
SOLVER_LS_ITERATIONS = 8

# The stock, CPU-matching setting, kept for the comparability cross-check.
STOCK_ITERATIONS = 100
STOCK_LS_ITERATIONS = 50


def make_model(solver_iterations=SOLVER_ITERATIONS,
               ls_iterations=SOLVER_LS_ITERATIONS):
    """MJX model at the approved 10/8 solver setting (pass 100/50 for the stock,
    CPU-matching dynamics)."""
    m = mujoco.MjModel.from_xml_path(XML)
    if solver_iterations:
        m.opt.iterations = solver_iterations
        m.opt.ls_iterations = ls_iterations
    return m


def observation(d):
    return jnp.concatenate([d.qpos[1:], jnp.clip(d.qvel, -10.0, 10.0)])


def is_healthy(d):
    z, ang = d.qpos[1], d.qpos[2]
    return ((z > HEALTHY_Z[0]) & (z < HEALTHY_Z[1]) &
            (ang > HEALTHY_ANGLE[0]) & (ang < HEALTHY_ANGLE[1]))


def make_env(mx):
    """-> (reset, step); both are pure, vmap-able over a batch of keys/states."""
    dt = mx.opt.timestep * FRAME_SKIP

    def reset(key):
        k1, k2, key = jax.random.split(key, 3)
        d = mjx.make_data(mx)
        d = d.replace(
            qpos=d.qpos + jax.random.uniform(k1, (mx.nq,), minval=-RESET_NOISE,
                                             maxval=RESET_NOISE),
            qvel=d.qvel + jax.random.uniform(k2, (mx.nv,), minval=-RESET_NOISE,
                                             maxval=RESET_NOISE))
        d = mjx.forward(mx, d)
        return State(data=d, obs=observation(d), reward=jnp.zeros(()),
                     done=jnp.zeros(()), steps=jnp.zeros((), jnp.int32), key=key)

    def step(state, action):
        action = jnp.clip(action, -1.0, 1.0)
        x_before = state.data.qpos[0]

        def phys(d, _):
            return mjx.step(mx, d.replace(ctrl=action)), None
        d, _ = jax.lax.scan(phys, state.data, None, length=FRAME_SKIP)

        x_after = d.qpos[0]
        healthy = is_healthy(d)
        reward = (HEALTHY_REWARD
                  + FORWARD_WEIGHT * (x_after - x_before) / dt
                  - CTRL_COST * jnp.sum(jnp.square(action)))
        steps = state.steps + 1
        terminated = ~healthy
        truncated = steps >= MAX_STEPS
        done = (terminated | truncated).astype(jnp.float32)

        # auto-reset so the batch never stalls: on done, resample a fresh episode.
        key, sub = jax.random.split(state.key)
        fresh = reset(sub)
        pick = lambda a, b: jax.tree.map(
            lambda x, y: jnp.where(done.astype(bool), x, y), a, b)
        nd = pick(fresh.data, d)
        nobs = jnp.where(done.astype(bool), fresh.obs, observation(d))
        nsteps = jnp.where(done.astype(bool), 0, steps).astype(jnp.int32)
        return State(data=nd, obs=nobs, reward=reward, done=done,
                     steps=nsteps, key=key)

    return reset, step
