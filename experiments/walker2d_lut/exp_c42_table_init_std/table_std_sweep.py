"""exp_c42 — choose the table init std by measuring the INITIAL POLICY, not the tensor.

The goal is stated in behavioural terms -- the initial policy should make SMALL, SMOOTH
movements -- so the sweep measures the policy, not the parameter. Both quantities are
computable at init with no training, on warmup-distribution states, in seconds.

WHAT IS MEASURED

  |mu|            mean absolute commanded action (post-tanh, which is what the actuator
                  receives). "Small" means near 0: the walker barely pushes.
  |mu| pre-tanh   the raw summed head output before the squash. Reported because tanh
                  compresses, so a large pre-tanh value can look deceptively moderate
                  after squashing while actually sitting in the saturated region where
                  gradients vanish.
  saturated       fraction of action components with |tanh(mu)| > 0.9. This is the real
                  danger of a large table init and the reason "small" matters beyond
                  aesthetics: a saturated tanh has almost no gradient, so those action
                  dimensions start nearly frozen.
  smooth-addr     mean |Delta action| between a state and the SAME state re-addressed one
                  cell away in one detector's digit. This is the smoothness that matters
                  for a LUT: neighbouring addresses should hold similar actions, otherwise
                  a one-bucket change in a single detector jerks the policy.
  smooth-time     mean |Delta action| between consecutive timesteps of a real rollout.
                  The behavioural read of the same thing.
  sigma           initial policy std, from the log-sigma half. Must stay ~exp(-1)=0.368 --
                  the trainer's -1/(heads*tph) bias is unchanged by this experiment and
                  this column exists to prove it.

Usage:
  python table_std_sweep.py
"""
import json
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(D, "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import mjx_walker2d as W                                    # noqa: E402
from mujoco import mjx                                      # noqa: E402
import jax_mhl_lut as LIF                                   # noqa: E402

OBS, ACT = 17, 6
HEADS, TPH, NDET, NB = 1, 32, 3, 4
DELAY_STD, BOUNDARY_OFFSET = 4.0, 0.0
N_TABLES, CELLS = HEADS * TPH, NB ** NDET
SEEDS = (0, 1, 2)
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0

FANIN = 0.1 / math.sqrt(TPH)         # 0.01768
STDS = [0.1, 0.05, FANIN, 0.01, 0.005, 0.002]


def rollout_states(n_steps=64):
    m = W.make_model()
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(jax.random.PRNGKey(12345), 64))

    @jax.jit
    def roll(st, key):
        def one(carry, _):
            st, key = carry
            key, k = jax.random.split(key)
            a = jax.random.uniform(k, (64, ACT), minval=-1.0, maxval=1.0)
            return (v_step(st, a), key), st.obs
        (st, key), obs = jax.lax.scan(one, (st, key), None, length=n_steps)
        return obs                                          # (T, 64, 17), time-ordered
    return roll(st, jax.random.PRNGKey(999))


def make_actor(seed, std):
    key = jax.random.PRNGKey(seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    p = LIF.init(ka, NB, NDET, TPH, HEADS, OBS, 2 * ACT, delay_init_std=DELAY_STD,
                 boundary_offset=BOUNDARY_OFFSET, table_init_std=std)
    return dict(p, table=p["table"].at[:, :, ACT:].add(-1.0 / (HEADS * TPH)))


def head(p, x):
    """(mu_pre_tanh, log_std) for normalised observations x."""
    y = LIF.apply(p, x, HEADS, TPH, NB, NDET, mode="eval").sum(1)
    return y[:, :ACT], jnp.clip(y[:, ACT:], LOGSTD_MIN, LOGSTD_MAX)


def measure(p, x_seq, x_flat):
    mu_pre, log_std = head(p, x_flat)
    act = np.asarray(jnp.tanh(mu_pre))
    mu_pre = np.asarray(mu_pre)

    # --- smoothness in ADDRESS space -------------------------------------
    # Re-read the same states with one detector's digit nudged by +1, and compare the
    # action. This isolates "do neighbouring addresses hold similar actions" from any
    # change in the states themselves.
    t_hard, t_soft = LIF.first_spike(p, x_flat)
    digits, _ = LIF.bucket(p, t_hard, t_soft)
    nudged = digits.at[:, :, 0].set(jnp.clip(digits[:, :, 0] + 1, 0, NB - 1))
    rows_a = LIF.hard_read(p, digits, NDET, NB).reshape(
        x_flat.shape[0], HEADS, TPH, -1).sum(2)
    rows_b = LIF.hard_read(p, nudged, NDET, NB).reshape(
        x_flat.shape[0], HEADS, TPH, -1).sum(2)
    a_a = np.asarray(jnp.tanh(rows_a[:, 0, :ACT]))
    a_b = np.asarray(jnp.tanh(rows_b[:, 0, :ACT]))
    smooth_addr = float(np.abs(a_a - a_b).mean())

    # --- smoothness in TIME ----------------------------------------------
    T, B, _ = x_seq.shape
    acts = np.asarray(jnp.tanh(head(p, x_seq.reshape(-1, OBS))[0])).reshape(T, B, ACT)
    smooth_time = float(np.abs(np.diff(acts, axis=0)).mean())

    return dict(abs_mu=float(np.abs(act).mean()),
                abs_mu_pre=float(np.abs(mu_pre).mean()),
                saturated=float((np.abs(act) > 0.9).mean()),
                smooth_addr=smooth_addr,
                smooth_time=smooth_time,
                sigma=float(np.exp(np.asarray(log_std)).mean()))


def main():
    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    raw = rollout_states()
    x_seq = (raw - om) / (osd + 1e-6)
    x_flat = x_seq.reshape(-1, OBS)
    print(f"{x_flat.shape[0]:,} warmup states ({x_seq.shape[0]} timesteps x "
          f"{x_seq.shape[1]} envs), averaged over seeds {SEEDS}")
    print(f"fan-in corrected std = 0.1/sqrt({TPH}) = {FANIN:.5f}\n")
    print(f"  {'table_std':>10}{'|mu|':>8}{'|mu| pre':>10}{'saturated':>11}"
          f"{'smooth-addr':>13}{'smooth-time':>13}{'sigma':>8}")

    rows = []
    for std in STDS:
        acc = [measure(make_actor(s, std), x_seq, x_flat) for s in SEEDS]
        m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        m["table_init_std"] = std
        rows.append(m)
        tag = "  <- stock" if std == 0.1 else ("  <- fan-in" if abs(std - FANIN) < 1e-9
                                               else "")
        print(f"  {std:>10.5f}{m['abs_mu']:>8.3f}{m['abs_mu_pre']:>10.3f}"
              f"{m['saturated']:>11.3f}{m['smooth_addr']:>13.4f}"
              f"{m['smooth_time']:>13.4f}{m['sigma']:>8.3f}{tag}")

    stock = rows[0]
    fan = next(r for r in rows if abs(r["table_init_std"] - FANIN) < 1e-9)
    print(f"\n  sigma is {min(r['sigma'] for r in rows):.3f}..."
          f"{max(r['sigma'] for r in rows):.3f} across the sweep — the trainer's "
          f"-1/(heads*tph) log-sigma bias is untouched, as intended (exp(-1) = 0.368)")
    print(f"  stock 0.1 -> fan-in {FANIN:.5f}:  |mu| {stock['abs_mu']:.3f} -> "
          f"{fan['abs_mu']:.3f}  ({100*fan['abs_mu']/stock['abs_mu']:.0f}% of stock), "
          f"saturated {stock['saturated']:.3f} -> {fan['saturated']:.3f}, "
          f"smooth-addr {stock['smooth_addr']:.4f} -> {fan['smooth_addr']:.4f}, "
          f"smooth-time {stock['smooth_time']:.4f} -> {fan['smooth_time']:.4f}")
    json.dump(dict(stds=rows, fanin=FANIN, stock=stock, chosen=fan),
              open(os.path.join(HERE, "table_std_sweep.json"), "w"), indent=1)
    print("\nwrote table_std_sweep.json")


if __name__ == "__main__":
    main()
