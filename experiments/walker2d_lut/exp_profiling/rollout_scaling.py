"""exp_profiling — is the rollout compute-bound or dispatch-bound?

The baseline profile shows the rollout costing ~51 ms per training iteration for ONE env
step over 64 envs, while the evaluator does 20,000 env steps (20 envs x 1,000) in 18.6 s
— 0.93 ms per step. A 55x discrepancy for the same physics cannot be physics.

The difference is structural: eval runs 1,000 steps inside a single `lax.scan` in one jit
call, while the training rollout is `scan(length=1)` and pays a full dispatch per call.

This measures it directly: time the same jitted rollout at scan lengths 1, 2, 4, 8, 16, 32.
If the cost is flat in the length, it is dispatch-bound and the per-iteration 51 ms is
almost entirely launch latency that could be amortised. If it scales linearly, it is real
physics and there is nothing to win.

Usage:
  python rollout_scaling.py
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import mjx_walker2d as W                                   # noqa: E402
from mujoco import mjx                                     # noqa: E402
import jax_bucket_lif as LIF                               # noqa: E402

OBS, ACT, ENVS, TPH, NB = 17, 6, 64, 32, 16
REPS = 30


def main():
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    key = jax.random.PRNGKey(0)
    key, ka, kr = jax.random.split(key, 3)
    p = LIF.init(ka, NB, TPH, 1, OBS, 2 * ACT)

    m = W.make_model()
    reset, step = W.make_env(mjx.put_model(m))
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(kr, ENVS))

    def make(length):
        @jax.jit
        def rollout(p, st, key):
            def one(carry, _):
                st, key = carry
                key, k1 = jax.random.split(key)
                x = (st.obs - om) / (osd + 1e-6)
                y = LIF.apply(p, x, 0.3, 1, TPH, NB, mode="st").sum(1)
                mu, ls = y[:, :ACT], jnp.clip(y[:, ACT:], -5.0, 2.0)
                act = jnp.tanh(mu + jnp.exp(ls) * jax.random.normal(k1, mu.shape))
                nst = v_step(st, act)
                return (nst, key), (st.obs, act, nst.reward, nst.obs, nst.done)
            (st, key), tr = jax.lax.scan(one, (st, key), None, length=length)
            return st, key, tr
        return rollout

    print(f"{'scan len':>9}{'ms/call':>10}{'ms/env-step':>14}{'vs len=1':>10}")
    base = None
    out = []
    for L in (1, 2, 4, 8, 16, 32):
        fn = make(L)
        r = fn(p, st, key)
        jax.block_until_ready(r)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = fn(p, st, key)
            jax.block_until_ready(r)
            ts.append(time.perf_counter() - t0)
        ms = 1000 * float(np.min(ts))
        base = base or ms
        print(f"{L:>9}{ms:>10.2f}{ms/L:>14.3f}{ms/base:>9.2f}x")
        out.append(dict(length=L, ms=ms, ms_per_step=ms / L))

    r = out[0]["ms"], out[-1]["ms"]
    print(f"\n  32 steps cost {r[1]/r[0]:.2f}x one step "
          f"(linear would be 32x).")
    print(f"  -> per-call FIXED overhead ~= {out[0]['ms'] - (out[-1]['ms']-out[0]['ms'])/31:.1f} ms, "
          f"marginal cost per env-step ~= {(out[-1]['ms']-out[0]['ms'])/31:.2f} ms")
    json.dump(out, open(os.path.join(HERE, "rollout_scaling.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
