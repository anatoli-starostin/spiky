"""exp_c32b — where do the spikes actually land, and what would quantile boundaries buy?

Groundwork for the Phase-3 hypothesis: *more buckets in a range the model never enters
buys nothing*. exp_c32's teardown showed the bucket index pinned in the top third of its
range; at 64 buckets the torch reference reaches only 21 of 64 at init. If the first-spike
distribution occupies a narrow band, then uniform boundaries over (0, t_window) spend most
of their resolution on empty time, and the fix is to place boundaries at QUANTILES of the
empirical spike-time distribution rather than uniformly.

This measures, per table:
  * the empirical distribution of t_hard on states the DEPLOYED policy visits
  * where the learned boundaries actually sit
  * the bucket occupancy entropy, in bits, against the log2(n_buckets) ceiling
  * the EFFECTIVE number of buckets in use, 2**entropy -- a fractional count that is not
    fooled by a bucket holding 0.1% of the mass, which a raw "distinct buckets" count is
  * what equal-mass (quantile) boundaries would give instead

The entropy comparison is the point. Equal-mass boundaries achieve the ceiling BY
CONSTRUCTION at the moment they are set, so the gap between the measured entropy and
log2(n_buckets) is exactly the headroom the reinitialisation could recover -- an upper
bound on the win, before training moves anything.

Usage:
  python spike_distribution.py <actor.npz> [--states 8192]
"""
import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--states", type=int, default=8192)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=400)
    a = ap.parse_args()
    path = a.actor if os.path.isabs(a.actor) else os.path.join(HERE, a.actor)

    z = np.load(path)
    p = {k: jnp.asarray(z[k]) for k in KEYS}
    heads, tph, nb = int(z["n_heads"]), int(z["tph"]), int(z["n_buckets"])
    n_tables = heads * tph
    print(f"=== {os.path.basename(path)}: spike-time distribution "
          f"({nb} buckets, {tph} tables) ===\n")

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    m = W.make_model()
    reset, step = W.make_env(mjx.put_model(m))
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(jax.random.PRNGKey(0), a.episodes))

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        return jnp.tanh(LIF.apply(p, x, 0.3, heads, tph, nb,
                                  mode="hard").sum(1)[:, :ACT])

    @jax.jit
    def run(st):
        def one(c, _):
            st, alive = c
            nst = v_step(st, act(st.obs))
            return (nst, alive * (1 - nst.done)), (st.obs, alive)
        (st, _), (o, al) = jax.lax.scan(one, (st, jnp.ones(a.episodes)), None,
                                        length=a.horizon)
        return o, al

    o, al = run(st)
    o = np.asarray(o).reshape(-1, OBS)[np.asarray(al).reshape(-1) > 0]
    idx = np.linspace(0, len(o) - 1, min(a.states, len(o))).astype(int)
    x = (jnp.asarray(o[idx]) - om) / (osd + 1e-6)
    print(f"  {x.shape[0]:,} states sampled from {len(o):,} visited\n")

    t_hard, _ = LIF.first_spike(p, x, n_tables)
    th = np.asarray(t_hard)                                # [S, T]
    bnd = np.asarray(LIF.boundaries(p))                    # [T, nb-1]
    addr = np.asarray(LIF.address(p, x, 0.3, heads, tph, nb))

    print("--- 1. where the spikes land ---")
    fin = th[th < 32.0]
    print(f"  t_hard over all (state, table): min {th.min():.2f}  "
          f"p1 {np.percentile(th,1):.2f}  p25 {np.percentile(th,25):.2f}  "
          f"median {np.median(th):.2f}  p75 {np.percentile(th,75):.2f}  "
          f"p99 {np.percentile(th,99):.2f}  max {th.max():.2f}")
    print(f"  no-spike (t = t_window = 32): {100*np.mean(th >= 32.0):.1f}%")
    print(f"  spiking mass spans [{fin.min():.2f}, {fin.max():.2f}] = "
          f"{100*(fin.max()-fin.min())/32.0:.0f}% of the 32-unit window")
    iqr = np.percentile(fin, 75) - np.percentile(fin, 25)
    print(f"  middle 50% of spikes span {iqr:.2f} time units "
          f"({100*iqr/32.0:.0f}% of the window)")

    print("\n--- 2. how the buckets are actually used ---")
    occ = np.zeros((n_tables, nb))
    for t in range(n_tables):
        occ[t] = np.bincount(addr[:, t], minlength=nb)
    pr = occ / occ.sum(1, keepdims=True)
    ent = -(np.where(pr > 0, pr * np.log2(np.maximum(pr, 1e-12)), 0)).sum(1)
    eff = 2 ** ent
    ceiling = np.log2(nb)
    print(f"  distinct buckets touched: median {int(np.median((occ>0).sum(1)))} "
          f"of {nb}   (a bucket with 0.1% of the mass counts here)")
    print(f"  occupancy entropy:  mean {ent.mean():.2f} bits of a possible "
          f"{ceiling:.2f}")
    print(f"  EFFECTIVE buckets used (2**entropy): mean {eff.mean():.1f} of {nb}  "
          f"[{eff.min():.1f}, {eff.max():.1f}]")
    print(f"  --> {100*(1 - eff.mean()/nb):.0f}% of the table's addressing capacity is "
          f"unused")
    print(f"  last-bucket (no-spike) mass: {100*np.mean(addr == nb-1):.1f}%")

    print("\n--- 3. what equal-mass (quantile) boundaries would give ---")
    q = np.linspace(0, 100, nb + 1)[1:-1]
    prop = np.stack([np.percentile(th[:, t], q) for t in range(n_tables)])
    # Equal-mass boundaries hit the entropy ceiling by construction at the moment they
    # are set. Report the headroom, and how far the current boundaries sit from them.
    print(f"  entropy now {ent.mean():.2f} bits -> quantile init {ceiling:.2f} bits "
          f"(by construction)")
    print(f"  effective buckets {eff.mean():.1f} -> {nb} "
          f"({nb/max(eff.mean(),1e-9):.1f}x more addressing resolution)")
    print(f"  current boundaries span   [{bnd.min():.2f}, {bnd.max():.2f}]")
    print(f"  quantile boundaries span  [{prop.min():.2f}, {prop.max():.2f}]")
    dup = float(np.mean(np.diff(prop, axis=-1) < 1e-3))
    print(f"  degenerate (zero-width) quantile gaps: {100*dup:.1f}%  "
          f"{'<-- would need jitter/merging' if dup > 0.02 else ''}")

    out = dict(actor=os.path.basename(path), n_buckets=nb, n_tables=n_tables,
               states=int(x.shape[0]), no_spike_frac=float(np.mean(th >= 32.0)),
               entropy_bits=float(ent.mean()), entropy_ceiling=float(ceiling),
               effective_buckets=float(eff.mean()),
               t_hard_percentiles={str(k): float(np.percentile(th, k))
                                   for k in (1, 5, 25, 50, 75, 95, 99)},
               proposed_boundaries=prop.tolist())
    o_path = path.replace("_actor.npz", "_spikedist.json")
    json.dump(out, open(o_path, "w"), indent=1)
    np.savez(path.replace("_actor.npz", "_spikedist.npz"),
             t_hard=th[:2048], boundaries=bnd, occupancy=occ, proposed=prop)
    print(f"\nwrote {os.path.basename(o_path)} (+ .npz)")


if __name__ == "__main__":
    main()
