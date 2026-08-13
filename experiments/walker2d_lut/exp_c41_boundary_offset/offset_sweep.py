"""exp_c41 — choose the per-detector BOUNDARY offset by measurement, not by guess.

Same cheap trick as exp_c40: everything that matters here is computable AT INIT with no
training, on 4,096 warmup-distribution states, in seconds. Sweeping is free, so the offset
is chosen from a curve rather than argued for.

WHAT IS DIFFERENT FROM THE DELAY OFFSET. A per-detector delay bias translates the whole
arrival pattern later, so past a few units the membrane stops crossing theta_mem and the
detector dies (34 of 96 dead at offset 5.33 in exp_c40). A per-detector BOUNDARY bias
cannot do that: the membrane, the arrivals and the spike time t* are untouched, and only
the ladder that reads t* moves. Dead-detector count is therefore expected to stay 0 across
the whole sweep, and that expectation is checked rather than assumed.

The quantiser can still SATURATE, which is a different failure: slide a ladder entirely
past the spike-time distribution and every sample lands in one bucket -- a constant digit,
zero entropy, and agreement back up. Boundaries start at [8, 16, 24] on a 32-wide window,
so both signs are swept out to where that must happen.

THE SUCCESS CRITERION IS EXCESS AGREEMENT, not raw agreement:

    excess = measured pairwise agreement - agreement that INDEPENDENT digits with the SAME
             marginals would give

The floor is not 1/n_buckets. The digit marginals are strongly non-uniform (no-spike mass
folds into the last bucket), which puts the floor near 0.50, not 0.25 -- an error I made in
the exp_c39 diagnosis and corrected in exp_c40. Only the excess means "redundant". Stock is
0.140 at init.

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
# The ladder can slide either way, so both signs are swept. Boundaries start at
# [8, 16, 24] on a 32-wide window, so |offset| past ~8 must saturate the quantiser.
OFFSETS = [-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0,
           1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]


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
    digits = np.asarray(LIF.bucket(p, t_hard, t_soft)[0])
    idx = np.asarray(LIF.cell_index(jnp.asarray(digits), NDET, NB))
    t_hard = np.asarray(t_hard)

    dc = np.stack([[(digits[:, t, d] == b).sum() for b in range(NB)]
                   for t in range(N_TABLES) for d in range(NDET)])
    det_ent = entropy_bits(dc.astype(np.float64))
    cc = np.stack([np.bincount(idx[:, t], minlength=CELLS) for t in range(N_TABLES)])
    eff = 2.0 ** entropy_bits(cc.astype(np.float64))

    agree, floor = [], []
    for t in range(N_TABLES):
        for d1 in range(NDET):
            for d2 in range(d1 + 1, NDET):
                u, v = digits[:, t, d1], digits[:, t, d2]
                agree.append((u == v).mean())
                pu = np.array([(u == k).mean() for k in range(NB)])
                pv = np.array([(v == k).mean() for k in range(NB)])
                floor.append(float((pu * pv).sum()))
    return dict(pair_agreement=float(np.mean(agree)),
                indep_floor=float(np.mean(floor)),
                det_entropy=float(det_ent.mean()),
                # "saturated", NOT "dead". A boundary shift cannot stop a neuron firing --
                # `nospike` below proves that, being invariant across the whole sweep.
                # Zero digit entropy here means the LADDER has slid off the spike-time
                # distribution so every sample lands in one bucket: a saturated quantiser
                # on a perfectly healthy detector. Conflating the two is exactly the
                # mistake this column is named to avoid.
                saturated=float((det_ent < 0.1).sum()),
                nospike=float((t_hard >= LIF.T_WINDOW).mean()),
                eff_cells=float(eff.mean()))


def main():
    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    x = (warmup_observations() - om) / (osd + 1e-6)
    print(f"{x.shape[0]:,} warmup-distribution states; {NDET} detectors x {NB} buckets; "
          f"averaged over seeds {SEEDS}")
    print("boundaries start at [8, 16, 24]; detector d's ladder slides by d*offset\n")
    print(f"  {'offset':>8}{'agree':>9}{'floor':>8}{'EXCESS':>9}{'entropy':>9}"
          f"{'satur':>7}{'nospike':>9}{'eff':>8}")

    rows = []
    for off in OFFSETS:
        acc = []
        for s in SEEDS:
            key = jax.random.PRNGKey(s)
            key, ka, kq, kr = jax.random.split(key, 4)
            p = LIF.init(ka, NB, NDET, TPH, HEADS, OBS, 2 * ACT,
                         delay_init_std=DELAY_STD, boundary_offset=off)
            p = dict(p, table=p["table"].at[:, :, ACT:].add(-1.0 / (HEADS * TPH)))
            acc.append(measure(p, x))
        m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        m["offset"] = off
        m["excess"] = m["pair_agreement"] - m["indep_floor"]
        rows.append(m)
        print(f"  {off:>8.1f}{m['pair_agreement']:>9.3f}{m['indep_floor']:>8.3f}"
              f"{m['excess']:>9.3f}{m['det_entropy']:>9.3f}{m['saturated']:>7.1f}"
              f"{m['nospike']:>9.3f}{m['eff_cells']:>8.2f}")

    base = next(r for r in rows if r["offset"] == 0.0)
    # THE STRUCTURAL CLAIM, checked rather than assumed: a boundary shift cannot change
    # whether a detector fires. If it holds, `nospike` is identical at every offset.
    nsp = [r["nospike"] for r in rows]
    print(f"\n  no-spike rate across the ENTIRE sweep: {min(nsp):.4f} .. {max(nsp):.4f}  "
          f"({'INVARIANT — firing is untouched, as designed' if max(nsp) - min(nsp) < 1e-6 else 'VARIES — the claim is wrong'})")
    print(f"  (contrast exp_c40's delay offset, where no-spike rose 0.483 -> 0.838 and "
          f"64 of 96 detectors stopped firing)")
    sat = [r for r in rows if r["saturated"] > 0.5]
    print(f"  offsets where the QUANTISER saturates (a healthy detector reading one "
          f"constant bucket): {[r['offset'] for r in sat]}")

    # Selection: the SMALLEST |offset| that reaches statistical independence. Once excess
    # is at zero there is nothing further to win, and sliding further only saturates.
    ok = [r for r in rows if r["saturated"] < 1.0 and r["det_entropy"] > 0.85]
    indep = [r for r in ok if r["excess"] <= 0.01]
    best = (min(indep, key=lambda r: abs(r["offset"])) if indep
            else (min(ok, key=lambda r: r["excess"]) if ok else None))
    print(f"  informative candidates (0 saturated, entropy > 0.85 bits): "
          f"{[r['offset'] for r in ok]}")
    print(f"  of those, statistically independent (excess <= 0.01): "
          f"{[r['offset'] for r in indep]} -> taking the smallest |offset|")
    if best:
        print(f"  CHOSEN offset {best['offset']:.1f}: EXCESS agreement "
              f"{base['excess']:.3f} -> {best['excess']:.3f}  (raw "
              f"{base['pair_agreement']:.3f} -> {best['pair_agreement']:.3f}, floor "
              f"{base['indep_floor']:.3f} -> {best['indep_floor']:.3f}); eff cells "
              f"{base['eff_cells']:.2f} -> {best['eff_cells']:.2f}; entropy "
              f"{base['det_entropy']:.3f} -> {best['det_entropy']:.3f}")
    json.dump(dict(offsets=rows, chosen=best, stock=base),
              open(os.path.join(HERE, "offset_sweep.json"), "w"), indent=1)
    print("\nwrote offset_sweep.json")


if __name__ == "__main__":
    main()
