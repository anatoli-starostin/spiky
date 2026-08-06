"""exp_c40 — choose the per-detector delay offset from measurement, not from a guess.

The whole point of the structured init is to break the within-table redundancy of the
detectors, measured at 0.62-0.64 pairwise digit agreement against a 0.25 floor for
independent uniform digits at 4 buckets. That quantity is computable AT INIT, with no
training at all, so the offset can be chosen by sweeping it in seconds instead of burning
sweeps.

MINIMISING AGREEMENT ALONE WOULD BE WRONG, and the sweep is built to show why. Pushing the
detectors far apart delays the later ones past the end of the window, their membranes never
cross theta_mem, and a non-firing detector folds into the LAST bucket -- a constant digit.
That drives agreement back UP and entropy to zero while looking superficially like
"spread". So four quantities are tracked together and the offset is chosen where agreement
is low while the detectors are still ALIVE:

    pair_agreement  within-table pairwise agreement of hard digits   -> want LOW (floor 0.25)
    det_entropy     per-detector digit entropy, bits (max 2 at M=4)  -> want HIGH
    nospike         fraction of detectors that never fire            -> want LOW
    eff_cells       2**entropy of per-table cell occupancy, of 64    -> want HIGH

Evaluated on warmup-distribution states (uniform random actions), the distribution the
model actually sees while the init still matters, and on 3 seeds so the choice is not made
on one draw.

Usage:
  python offset_sweep.py
"""
import json
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
DELAY_STD = 4.0
N_TABLES, CELLS = HEADS * TPH, NB ** NDET
SEEDS = (0, 1, 2)
# t_window/(2D) = 5.33 is the "tile the window in D phases" value; the sweep brackets it
# generously on both sides so the shape of the curve is visible, not just one point.
OFFSETS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.333, 6.0, 8.0, 10.667, 13.0, 16.0]


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


def entropy_bits(counts, axis=-1):
    tot = counts.sum(axis, keepdims=True)
    q = np.where(tot > 0, counts / np.maximum(tot, 1e-12), 0.0)
    return -(q * np.log2(np.where(q > 0, q, 1.0))).sum(axis)


def measure(p, x):
    t_hard, t_soft = LIF.first_spike(p, x)
    digits, _ = LIF.bucket(p, t_hard, t_soft)
    idx = np.asarray(LIF.cell_index(digits, NDET, NB))
    digits, t_hard = np.asarray(digits), np.asarray(t_hard)

    dc = np.stack([[(digits[:, t, d] == b).sum() for b in range(NB)]
                   for t in range(N_TABLES) for d in range(NDET)])
    det_ent = entropy_bits(dc.astype(np.float64))
    cc = np.stack([np.bincount(idx[:, t], minlength=CELLS) for t in range(N_TABLES)])
    eff = 2.0 ** entropy_bits(cc.astype(np.float64))

    agree, indep = [], []
    for t in range(N_TABLES):
        for d1 in range(NDET):
            for d2 in range(d1 + 1, NDET):
                a, b = digits[:, t, d1], digits[:, t, d2]
                agree.append((a == b).mean())
                # agreement expected if the two digits were INDEPENDENT with these same
                # marginals -- the honest floor, since the marginals are not uniform
                pa = np.array([(a == k).mean() for k in range(NB)])
                pb = np.array([(b == k).mean() for k in range(NB)])
                indep.append(float((pa * pb).sum()))
    return dict(pair_agreement=float(np.mean(agree)),
                indep_floor=float(np.mean(indep)),
                det_entropy=float(det_ent.mean()),
                det_entropy_min=float(det_ent.min()),
                dead=int((det_ent < 0.1).sum()),
                nospike=float((t_hard >= LIF.T_WINDOW).mean()),
                eff_cells=float(eff.mean()))


def main():
    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    x = (warmup_observations() - om) / (osd + 1e-6)
    print(f"{x.shape[0]:,} warmup-distribution states; {NDET} detectors x {NB} buckets; "
          f"averaged over seeds {SEEDS}\n")
    print(f"  {'offset':>8}{'agree':>9}{'floor':>8}{'entropy':>9}{'min-ent':>9}"
          f"{'dead':>6}{'nospike':>9}{'eff':>8}")

    rows = []
    for off in OFFSETS:
        acc = []
        for s in SEEDS:
            key = jax.random.PRNGKey(s)
            key, ka, kq, kr = jax.random.split(key, 4)
            p = LIF.init(ka, NB, NDET, TPH, HEADS, OBS, 2 * ACT,
                         delay_init_std=DELAY_STD, delay_offset=off)
            p["table"] = p["table"].at[:, :, ACT:].add(-1.0 / (HEADS * TPH))
            acc.append(measure(p, x))
        m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        m["offset"] = off
        rows.append(m)
        print(f"  {off:>8.3f}{m['pair_agreement']:>9.3f}{m['indep_floor']:>8.3f}"
              f"{m['det_entropy']:>9.3f}{m['det_entropy_min']:>9.3f}"
              f"{m['dead']:>6.1f}{m['nospike']:>9.3f}{m['eff_cells']:>8.2f}")

    # Pick: lowest agreement among offsets that keep the detectors alive.
    ok = [r for r in rows if r["dead"] < 1.0 and r["det_entropy"] > 0.85]
    best = min(ok, key=lambda r: r["pair_agreement"]) if ok else None
    print(f"\n  alive-and-informative candidates (0 dead detectors, mean entropy > 0.85 "
          f"bits): {[r['offset'] for r in ok]}")
    if best:
        base = rows[0]
        print(f"  CHOSEN offset {best['offset']:.3f}: pair agreement "
              f"{base['pair_agreement']:.3f} -> {best['pair_agreement']:.3f} "
              f"(independent floor {best['indep_floor']:.3f}), "
              f"eff cells {base['eff_cells']:.2f} -> {best['eff_cells']:.2f}, "
              f"no-spike {base['nospike']:.3f} -> {best['nospike']:.3f}")
    json.dump(dict(offsets=rows, chosen=best),
              open(os.path.join(HERE, "offset_sweep.json"), "w"), indent=1)
    print("\nwrote offset_sweep.json")


if __name__ == "__main__":
    main()
