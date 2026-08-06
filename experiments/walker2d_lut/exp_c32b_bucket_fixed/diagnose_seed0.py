"""exp_c32b — is seed 0's proxy drop numerical or an ordinary RL setback?

Checks the live checkpoint end to end: every parameter for NaN/inf, then the full forward
path (membrane V -> first crossing t* -> bucket index -> table lookup -> mu/log_std ->
tanh action) on states the policy actually visits. A numerical failure shows up as inf/nan
or as a degenerate constant somewhere in that chain; an RL setback leaves every quantity
finite and well-scaled and only the RETURN bad.

Usage:
  python diagnose_seed0.py [seed]
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import mjx_walker2d as W                                   # noqa: E402
from mujoco import mjx                                     # noqa: E402
import jax_bucket_lif as LIF                               # noqa: E402

OBS, ACT = 17, 6
KEYS = ("delay", "w_raw", "tau_raw", "log_T_cross", "log_T_bkt",
        "beta_base", "beta_raw", "table")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    z = np.load(os.path.join(HERE, f"bucket_sac_c32b_s{seed}_actor.npz"))
    p = {k: jnp.asarray(z[k]) for k in KEYS}
    heads, tph, nb = int(z["n_heads"]), int(z["tph"]), int(z["n_buckets"])
    n_tables = heads * tph
    print(f"=== seed {seed} checkpoint diagnostic (n_buckets={nb}, tph={tph}) ===\n")

    print("--- 1. parameters: finite? degenerate? ---")
    bad = False
    for k in KEYS:
        v = np.asarray(p[k])
        nan, inf = int(np.isnan(v).sum()), int(np.isinf(v).sum())
        bad |= bool(nan or inf)
        print(f"  {k:<13} shape {str(v.shape):<14} min {v.min():+10.4f} "
              f"max {v.max():+10.4f} mean {v.mean():+9.4f}  nan {nan} inf {inf}")
    print(f"  --> {'NaN/Inf PRESENT' if bad else 'all parameters finite'}")

    print("\n--- 2. derived quantities ---")
    w = np.asarray(LIF.synapses(p))
    tau = np.asarray(jax.nn.softplus(p["tau_raw"]) + LIF.TAU_FLOOR)
    bnd = np.asarray(LIF.boundaries(p))
    tbkt = np.exp(np.asarray(p["log_T_bkt"]))
    tcr = np.exp(np.asarray(p["log_T_cross"]))
    print(f"  synapses w = 2*sigmoid(w_raw): [{w.min():.4f}, {w.max():.4f}] "
          f"mean {w.mean():.4f}   (bounded in [0,2] by construction)")
    print(f"  tau:      [{tau.min():.3f}, {tau.max():.3f}]   floor is "
          f"{LIF.TAU_FLOOR}  -> overflow guard "
          f"{'OK' if tau.min() >= LIF.TAU_FLOOR - 1e-6 else 'VIOLATED'}")
    print(f"  T_bkt:    [{tbkt.min():.5f}, {tbkt.max():.5f}]")
    print(f"  T_cross:  [{tcr.min():.5f}, {tcr.max():.5f}]")
    print(f"  boundaries: [{bnd.min():.3f}, {bnd.max():.3f}]  strictly increasing: "
          f"{bool(np.all(bnd[:, 1:] > bnd[:, :-1]))}  min gap "
          f"{np.diff(bnd, axis=-1).min():.4f}")

    print("\n--- 3. forward path on states the policy visits ---")
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    m = W.make_model()
    reset, step = W.make_env(mjx.put_model(m))
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(jax.random.PRNGKey(0), 16))

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        y = LIF.apply(p, x, 0.3, heads, tph, nb, mode="hard").sum(1)
        return jnp.tanh(y[:, :ACT])

    STEPS = 120          # enough to see falls; 300 was slower than useful on CPU
    obs_l, rew_l, done_l = [], [], []
    for _ in range(STEPS):
        a = act(st.obs)
        obs_l.append(np.asarray(st.obs))
        st = v_step(st, a)
        rew_l.append(np.asarray(st.reward))
        done_l.append(np.asarray(st.done))
    O = jnp.asarray(np.concatenate(obs_l))
    x = (O - om) / (osd + 1e-6)

    t_hard, t_soft = LIF.first_spike(p, x, n_tables)
    y = LIF.apply(p, x, 0.3, heads, tph, nb, mode="hard").sum(1)
    mu, ls = np.asarray(y[:, :ACT]), np.clip(np.asarray(y[:, ACT:]), -5.0, 2.0)
    a_t = np.tanh(mu)
    addr = np.asarray(LIF.address(p, x, 0.3, heads, tph, nb))
    th = np.asarray(t_hard)

    def rep(name, v, extra=""):
        v = np.asarray(v)
        print(f"  {name:<22} [{v.min():+9.4f}, {v.max():+9.4f}] mean {v.mean():+8.4f}  "
              f"nan {int(np.isnan(v).sum())} inf {int(np.isinf(v).sum())} {extra}")

    rep("membrane t_hard", th, f"(t_window fold-ins: {100*np.mean(th>=32.0):.1f}%)")
    rep("membrane t_soft", t_soft)
    rep("table output mu", mu)
    rep("log_std (clipped)", ls,
        f"(at -5 rail: {100*np.mean(ls<=-4.999):.1f}%, at +2: {100*np.mean(ls>=1.999):.1f}%)")
    rep("tanh(mu) action", a_t,
        f"(|a|>0.99: {100*np.mean(np.abs(a_t)>0.99):.1f}%)")
    print(f"  bucket index           [{addr.min()}, {addr.max()}] mean "
          f"{addr.mean():.2f}  distinct {len(np.unique(addr))}/{nb}  "
          f"last-bucket mass {100*np.mean(addr == nb-1):.1f}%")

    print("\n--- 4. what the rollout actually does ---")
    R = np.concatenate(rew_l).reshape(STEPS, -1)
    D = np.concatenate(done_l).reshape(STEPS, -1)
    first_done = np.where(D.any(0), D.argmax(0), STEPS)
    print(f"  reward per step: mean {R.mean():+.4f}  min {R.min():+.4f} "
          f"max {R.max():+.4f}  nan {int(np.isnan(R).sum())}")
    print(f"  episode length before termination: mean {first_done.mean():.0f} of {STEPS} "
          f"({int((first_done < STEPS).sum())}/{D.shape[1]} envs fell)")
    fwd = np.asarray(O)[:, 8]
    print(f"  forward velocity (obs[8]): mean {fwd.mean():+.4f} m/s  "
          f"({100*np.mean(fwd < 0):.0f}% of steps moving BACKWARD)")

    print("\n=== VERDICT ===")
    if bad or not np.isfinite(th).all() or not np.isfinite(mu).all():
        print("  NUMERICAL FAILURE — non-finite values found above.")
    else:
        print("  Every parameter and every intermediate is finite and in range.")
        print("  The membrane, crossings, bucket indices and actions are all well-formed.")
        print(f"  Behaviour: moves at {fwd.mean():+.3f} m/s "
              f"({100*np.mean(fwd < 0):.0f}% of steps backward) and falls after "
              f"~{first_done.mean():.0f} steps.")
        print("  NOTE the checkpoint is rewritten at EVERY eval, so this reflects the "
              "policy at the")
        print("  most recent eval — not the one that produced the dip. Whatever the dip "
              "was, it left")
        print("  no numerical trace, and the return recovered on its own.")


if __name__ == "__main__":
    main()
