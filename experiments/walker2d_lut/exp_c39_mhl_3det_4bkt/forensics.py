"""exp_c39 diagnosis, phases 2-3 — init forensics and final-state forensics.

INIT IS NOT LOST. The trainer never saved its starting state, but it does not need to have:
it is deterministic from `jax.random.PRNGKey(seed)` and the split order is fixed, so every
seed's exact init is reproducible here. This regenerates it rather than reasoning about
what it probably looked like.

WHAT IS AND IS NOT SEED-DEPENDENT AT INIT, which decides what can possibly be an init-level
predictor. `tau_raw`, `beta_base`, `beta_raw`, `log_T_cross` and `log_T_bkt` are all
CONSTANTS at init -- so the three boundaries per detector sit at exactly 8, 16, 24 for every
detector of every table of every seed, and "boundaries bunched so a detector is effectively
constant" cannot be a seed-level explanation at step 0. Only `delay` (half-normal, scale 4),
`w_raw` (N(-2.2, 0.5)) and `table` (0.1*randn) differ between seeds. So the init question
reduces to: do the delays and synapses of one seed produce a measurably different ADDRESSING
FUNCTION than another's?

That has to be measured functionally, on observations, not from the raw tensors -- two
seeds' weights are drawn from the same distribution by construction, so any difference lives
in what they DO, not in their summary statistics.

THE OBSERVATION SET is a warmup-distribution rollout: uniform random actions, the exact
distribution the model sees for its first 500 iterations, when init is the only thing that
distinguishes the seeds. It is generated ONCE from a fixed key and reused for every seed, so
all comparisons are on identical inputs.

FINAL-STATE FORENSICS answers the question the addressing diagnostics could not: is failure
"the detectors never activate" or "the detectors activate but the table never learns"?
Because init is reproducible, `table_final - table_init` is computable exactly, per row --
so we can see which rows moved, by how much, and whether the rows that moved are the rows
that get addressed.

Usage:
  python forensics.py
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
TOOK_OFF = {0: False, 1: False, 2: True}
FINAL = {0: 890.8, 1: 982.3, 2: 4217.3}
N_OBS_STEPS = 64                    # x 64 envs = 4,096 observations


def actor_init(key):
    """Byte-for-byte the trainer's actor_init."""
    p = LIF.init(key, NB, NDET, TPH, HEADS, OBS, 2 * ACT, delay_init_std=DELAY_STD)
    p["table"] = p["table"].at[:, :, ACT:].add(-1.0 / (HEADS * TPH))
    return p


def init_for_seed(seed):
    """The trainer's key schedule: PRNGKey(seed) -> split 4 -> ka is the actor's."""
    key = jax.random.PRNGKey(seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    return actor_init(ka)


def warmup_observations():
    """Uniform-random-action rollout: the distribution the model sees during warmup.

    One fixed key, reused for every seed, so all seeds are compared on identical inputs."""
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
            nst = v_step(st, a)
            return (nst, key), st.obs
        (st, key), obs = jax.lax.scan(one, (st, key), None, length=N_OBS_STEPS)
        return obs.reshape(-1, OBS)
    return roll(st, jax.random.PRNGKey(999))


def entropy_bits(counts, axis=-1):
    tot = counts.sum(axis, keepdims=True)
    q = np.where(tot > 0, counts / np.maximum(tot, 1e-12), 0.0)
    return -(q * np.log2(np.where(q > 0, q, 1.0))).sum(axis)


def address_stats(p, x):
    """Everything measurable about how a parameter set ADDRESSES a given input set."""
    t_hard, t_soft = LIF.first_spike(p, x)
    digits, _ = LIF.bucket(p, t_hard, t_soft)                    # (B,T,D) int
    idx = np.asarray(LIF.cell_index(digits, NDET, NB))           # (B,T)
    digits = np.asarray(digits)
    t_hard = np.asarray(t_hard)

    # per-detector digit entropy, in bits, max log2(NB)
    dc = np.stack([[(digits[:, t, d] == b).sum() for b in range(NB)]
                   for t in range(N_TABLES) for d in range(NDET)])
    det_ent = entropy_bits(dc.astype(np.float64)).reshape(N_TABLES, NDET)

    # per-table cell occupancy -> effective cells
    cc = np.stack([np.bincount(idx[:, t], minlength=CELLS) for t in range(N_TABLES)])
    eff = 2.0 ** entropy_bits(cc.astype(np.float64))

    # detector redundancy: pairwise agreement of digits WITHIN a table. Two detectors that
    # always agree collapse the mixed-radix product -- 4**3 cells becomes 4**2 in effect.
    agree = []
    for t in range(N_TABLES):
        for d1 in range(NDET):
            for d2 in range(d1 + 1, NDET):
                agree.append((digits[:, t, d1] == digits[:, t, d2]).mean())
    agree = np.asarray(agree)

    return dict(
        nospike=float((t_hard >= LIF.T_WINDOW).mean()),
        det_entropy_mean=float(det_ent.mean()),
        det_entropy_min=float(det_ent.min()),
        dead_detectors=int((det_ent < 0.1).sum()),          # of N_TABLES*NDET
        live_detectors=int((det_ent > 0.5).sum()),
        eff_cells_mean=float(eff.mean()),
        eff_cells_min=float(eff.min()),
        cells_touched_mean=float((cc > 0).sum(-1).mean()),
        pair_agreement_mean=float(agree.mean()),
        pair_agreement_max=float(agree.max()),
        digit_mean=float(digits.mean()),
        occupancy=cc,
    )


def main():
    x_raw = warmup_observations()
    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    x = (x_raw - om) / (osd + 1e-6)
    print(f"observation set: {x.shape[0]:,} warmup-distribution states "
          f"(uniform random actions, one fixed key, identical for every seed)\n")

    out = {}
    for s in SEEDS:
        p0 = init_for_seed(s)
        z = np.load(os.path.join(HERE, f"mhl_sac_c39_s{s}_actor.npz"))
        pf = {k: jnp.asarray(z[k]) for k in
              ("delay", "w_raw", "tau_raw", "beta_base", "beta_raw",
               "log_T_cross", "log_T_bkt", "table")}

        a0 = address_stats(p0, x)
        af = address_stats(pf, x)

        # --- how far did the table actually move, row by row? ---------------
        t0 = np.asarray(p0["table"])
        tf = np.asarray(pf["table"])
        disp = np.linalg.norm(tf - t0, axis=-1)                  # (T, cells)
        occ_f = af["occupancy"]                                  # (T, cells)
        addressed = occ_f > 0
        # Rows the FINAL policy actually uses, vs rows it never visits.
        d_used = disp[addressed]
        d_unused = disp[~addressed]
        # Is displacement concentrated where the addressing is? Spearman-free check:
        # displacement of the top-decile-occupancy rows vs the rest.
        flat_occ, flat_disp = occ_f.ravel(), disp.ravel()
        k = max(1, int(0.1 * flat_occ.size))
        top = np.argsort(flat_occ)[-k:]
        out[s] = dict(
            took_off=TOOK_OFF[s], final_cpu_ref=FINAL[s],
            init={k2: v for k2, v in a0.items() if k2 != "occupancy"},
            final={k2: v for k2, v in af.items() if k2 != "occupancy"},
            table_disp_mean=float(disp.mean()),
            table_disp_used_mean=float(d_used.mean()) if d_used.size else 0.0,
            table_disp_unused_mean=float(d_unused.mean()) if d_unused.size else 0.0,
            table_disp_top_decile=float(flat_disp[top].mean()),
            table_disp_max=float(disp.max()),
            rows_moved_gt_0p1=int((disp > 0.1).sum()),
            rows_addressed_final=int(addressed.sum()),
            delay_mean=float(np.asarray(p0["delay"]).mean()),
            delay_std=float(np.asarray(p0["delay"]).std()),
            w_init_mean=float(np.asarray(LIF.synapses(p0)).mean()),
        )

    # ---------------- report -------------------------------------------------
    def row(title, key, path, fmt="{:.3f}"):
        vals = []
        for s in SEEDS:
            v = out[s]
            for pp in path:
                v = v[pp]
            vals.append(v)
        cells = "".join(f"{fmt.format(v):>12}" for v in vals)
        print(f"  {title:<34}{cells}")

    hdr = "".join(f"{('s'+str(s)+(' WIN' if TOOK_OFF[s] else ' flat')):>12}"
                  for s in SEEDS)
    print("=== INIT (regenerated exactly from PRNGKey(seed)) ===")
    print(f"  {'':<34}{hdr}")
    row("delay mean", None, [ "delay_mean"])
    row("delay std", None, ["delay_std"])
    row("effective w mean", None, ["w_init_mean"])
    row("no-spike rate", None, ["init", "nospike"])
    row("detector digit entropy (bits)", None, ["init", "det_entropy_mean"])
    row("  worst detector", None, ["init", "det_entropy_min"])
    row("dead detectors (<0.1 bit) of 96", None, ["init", "dead_detectors"], "{:.0f}")
    row("effective cells / table", None, ["init", "eff_cells_mean"])
    row("cells touched / table (of 64)", None, ["init", "cells_touched_mean"])
    row("detector pair agreement", None, ["init", "pair_agreement_mean"])
    row("  worst pair", None, ["init", "pair_agreement_max"])

    print("\n=== FINAL (same observation set, so addressing is comparable) ===")
    print(f"  {'':<34}{hdr}")
    row("no-spike rate", None, ["final", "nospike"])
    row("detector digit entropy (bits)", None, ["final", "det_entropy_mean"])
    row("dead detectors (<0.1 bit) of 96", None, ["final", "dead_detectors"], "{:.0f}")
    row("live detectors (>0.5 bit) of 96", None, ["final", "live_detectors"], "{:.0f}")
    row("effective cells / table", None, ["final", "eff_cells_mean"])
    row("cells touched / table (of 64)", None, ["final", "cells_touched_mean"])
    row("detector pair agreement", None, ["final", "pair_agreement_mean"])

    print("\n=== TABLE LEARNING (||final - init|| per row) ===")
    print(f"  {'':<34}{hdr}")
    row("mean row displacement", None, ["table_disp_mean"])
    row("  rows the policy addresses", None, ["table_disp_used_mean"])
    row("  rows it never addresses", None, ["table_disp_unused_mean"])
    row("  top-decile-occupancy rows", None, ["table_disp_top_decile"])
    row("max row displacement", None, ["table_disp_max"])
    row("rows moved >0.1 (of 2048)", None, ["rows_moved_gt_0p1"], "{:.0f}")
    row("rows addressed at all (of 2048)", None, ["rows_addressed_final"], "{:.0f}")
    print(f"\n  final CPU reference{'':<16}" +
          "".join(f"{FINAL[s]:>12.0f}" for s in SEEDS))

    json.dump(out, open(os.path.join(HERE, "forensics.json"), "w"), indent=1,
              default=float)
    print("\nwrote forensics.json")


if __name__ == "__main__":
    main()
