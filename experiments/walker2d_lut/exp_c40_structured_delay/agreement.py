"""exp_c40 — within-table detector pair agreement, before and after training, c39 vs c40.

The structured init exists to break the redundancy between the D detectors of a table. This
measures whether it did, at init AND at the end, for both experiments on one common
observation set.

Reported against the INDEPENDENCE FLOOR computed from the digits' own marginals, not
against 1/n_buckets. That distinction matters and I got it wrong the first time: with the
no-spike mass folding into the last bucket the marginals are far from uniform, so the floor
for independent digits is ~0.50, not 0.25. Agreement of 0.63 against a 0.50 floor is a much
smaller redundancy than 0.63 against 0.25.

Usage:
  python agreement.py
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C39 = os.path.join(D, "exp_c39_mhl_3det_4bkt")
sys.path.insert(0, os.path.join(D, "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import mjx_walker2d as W                                    # noqa: E402
from mujoco import mjx                                      # noqa: E402
import jax_mhl_lut as LIF                                   # noqa: E402

OBS, ACT = 17, 6
HEADS, TPH, NDET, NB = 1, 32, 3, 4
N_TABLES = HEADS * TPH
PKEYS = ("delay", "w_raw", "tau_raw", "beta_base", "beta_raw",
         "log_T_cross", "log_T_bkt", "table")


def warmup_observations():
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
        (st, key), obs = jax.lax.scan(one, (st, key), None, length=64)
        return obs.reshape(-1, OBS)
    return roll(st, jax.random.PRNGKey(999))


def agree(p, x):
    t_hard, t_soft = LIF.first_spike(p, x)
    digits = np.asarray(LIF.bucket(p, t_hard, t_soft)[0])
    a_, f_ = [], []
    for t in range(N_TABLES):
        for d1 in range(NDET):
            for d2 in range(d1 + 1, NDET):
                u, v = digits[:, t, d1], digits[:, t, d2]
                a_.append((u == v).mean())
                pu = np.array([(u == k).mean() for k in range(NB)])
                pv = np.array([(v == k).mean() for k in range(NB)])
                f_.append(float((pu * pv).sum()))
    return float(np.mean(a_)), float(np.mean(f_))


def init_for(seed, offset):
    key = jax.random.PRNGKey(seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    p = LIF.init(ka, NB, NDET, TPH, HEADS, OBS, 2 * ACT,
                 delay_init_std=4.0, delay_offset=offset)
    return dict(p, table=p["table"].at[:, :, ACT:].add(-1.0 / (HEADS * TPH)))


def main():
    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    x = (warmup_observations() - om) / (osd + 1e-6)
    print(f"{x.shape[0]:,} warmup-distribution states, identical for every row\n")
    print(f"  {'':<26}{'agreement':>11}{'indep floor':>13}{'excess':>9}")

    out = {}
    for tag, folder, offset in (("c39 stock (offset 0)", C39, 0.0),
                                ("c40 structured (offset 2)", HERE, 2.0)):
        ia, ifl, fa, ffl = [], [], [], []
        for s in (0, 1, 2):
            a0, f0 = agree(init_for(s, offset), x)
            ia.append(a0)
            ifl.append(f0)
            stem = "c39" if "c39" in tag else "c40"
            z = np.load(os.path.join(folder, f"mhl_sac_{stem}_s{s}_actor.npz"))
            pf = {k: jnp.asarray(z[k]) for k in PKEYS}
            a1, f1 = agree(pf, x)
            fa.append(a1)
            ffl.append(f1)
        for lab, a_, f_ in ((f"{tag} INIT", ia, ifl), (f"{tag} FINAL", fa, ffl)):
            am, fm = float(np.mean(a_)), float(np.mean(f_))
            print(f"  {lab:<26}{am:>11.3f}{fm:>13.3f}{am - fm:>9.3f}")
        out[tag] = dict(init_agreement=float(np.mean(ia)),
                        init_floor=float(np.mean(ifl)),
                        final_agreement=float(np.mean(fa)),
                        final_floor=float(np.mean(ffl)))

    print("\n  'excess' = agreement above what independent digits with the SAME marginals "
          "would give.\n  That is the only number that means 'redundant'.")
    json.dump(out, open(os.path.join(HERE, "agreement.json"), "w"), indent=1)
    print("\nwrote agreement.json")


if __name__ == "__main__":
    main()
