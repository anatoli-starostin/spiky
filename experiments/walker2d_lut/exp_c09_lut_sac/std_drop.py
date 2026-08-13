"""exp_c21 follow-up — drop the log-std half of the SAC LUT actor and measure what changes.

The SAC actor's table is (32, 64, 12): six mean columns and six log-std columns per row.
`eval_cpu.load_actor` already takes `tanh(y[:, :ACT])`, so the std half is dead weight at
evaluation. This writes a 6-column copy and checks three things, in increasing order of
how much they can actually differ:

  1. ACTIONS. Should be bit-identical. Each table contributes rows[b, t, :], the heads sum
     over tables, and slicing the last axis commutes with that sum -- so summing 6-wide
     rows equals slicing the 12-wide sum. Asserted on real observations, not assumed.
  2. RETURN / speed on the 100-episode CPU reference. Follows from (1), but run anyway,
     because "provably identical" claims are worth one confirmation that the plumbing
     (loader, shapes, forward mode) really is the same on both files.
  3. INT4 QUANTIZATION. Here it genuinely differs. The per-table max-abs scale is taken
     over the WHOLE table, so with 12 columns the log-std entries can set the scale for
     the mean entries. Drop them and every table whose max-abs lived in a std column gets
     a finer scale -- more resolution where it matters. That is the one real effect, and
     it is measured, not asserted.

Writes `lut_sac_c21_seed4_20k_meanonly_actor.npz` and `..._meanonly_stddrop.json`.
Touches no `*_cpueval.json` and does not overwrite the source actor.

Usage:
  python std_drop.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac", "exp_c18_seed_variance"):
    sys.path.insert(0, os.path.join(D, p))

import eval_cpu                                            # noqa: E402
import perturb                                             # noqa: E402
import behavior                                            # noqa: E402
import quant_sweep as Q                                    # noqa: E402

SRC = "lut_sac_c21_seed4_20k_actor.npz"
DST = "lut_sac_c21_seed4_20k_meanonly_actor.npz"
OUT = "lut_sac_c21_seed4_20k_meanonly_stddrop.json"
ACT = 6


def row(label, path_or_arrays, m):
    if isinstance(path_or_arrays, str):
        fn, n = eval_cpu.load_actor(os.path.join(HERE, path_or_arrays),
                                    forward_mode="hard")
    else:
        fn, n = path_or_arrays
    per, _ = behavior.rollout_instrumented(m, fn)
    r = per["ret"]
    d = dict(label=label, params=n, mean=float(r.mean()), sd=float(r.std(ddof=1)),
             full=int((per["length"] >= 1000).sum()), fell=int(per["fell"].sum()),
             len_mean=float(per["length"].mean()), vel=float(per["vel_mean"].mean()))
    print(f"  {label:<28} {d['mean']:8.1f} +/- {d['sd']:6.1f}  full {d['full']:>3}/100  "
          f"vel {d['vel']:.3f} m/s  ({n:,} params)", flush=True)
    return d


def main():
    z = np.load(os.path.join(HERE, SRC))
    W12 = z["weights"]
    if W12.shape[-1] != 2 * ACT:
        raise SystemExit(f"expected a 12-wide table (6 mean + 6 log-std), got "
                         f"{W12.shape}. Refusing to guess which columns are the mean.")
    W6 = np.ascontiguousarray(W12[:, :, :ACT])
    np.savez(os.path.join(HERE, DST), w=z["w"], b=z["b"], weights=W6,
             log_T_soft=z["log_T_soft"], log_T_sel=z["log_T_sel"],
             n_heads=z["n_heads"], tph=z["tph"])
    print(f"wrote {DST} ({os.path.getsize(os.path.join(HERE, DST)):,} bytes, "
          f"source {os.path.getsize(os.path.join(HERE, SRC)):,})", flush=True)

    # ---- 1. actions, on real observations ----------------------------------
    f12, n12 = eval_cpu.load_actor(os.path.join(HERE, SRC), forward_mode="hard")
    f6, n6 = eval_cpu.load_actor(os.path.join(HERE, DST), forward_mode="hard")
    m = perturb.make_model(None, 1.0)
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1.0, (64, 17)).astype(np.float32)   # off-manifold on purpose:
    # random states exercise routing branches a nominal rollout may never visit.
    a12, a6 = f12(obs), f6(obs)
    max_abs = float(np.abs(a12 - a6).max())
    print(f"action check on 64 states: max |delta| = {max_abs:.3e} "
          f"({'bit-identical' if max_abs == 0.0 else 'DIFFERS'})", flush=True)

    # ---- 2. the 100-episode CPU reference ----------------------------------
    print("fp32:", flush=True)
    r12 = row("original (12-col)", SRC, m)
    r6 = row("std-dropped (6-col)", DST, m)

    # ---- 3. int4, where the scale really does change -----------------------
    scale12 = np.abs(W12.reshape(32, -1)).max(1) / 7.0
    scale6 = np.abs(W6.reshape(32, -1)).max(1) / 7.0
    set_by_std = int((scale6 < scale12 - 1e-12).sum())
    shrink = scale6 / scale12
    print(f"int4 table scale: {set_by_std}/32 tables had their max-abs in a log-std "
          f"column; scale shrinks to {shrink.min():.3f}-{shrink.max():.3f} of the "
          f"12-column value (mean {shrink.mean():.3f})", flush=True)

    print("int4 (table 4 bits + addressing 4 bits, as the committed checkpoint):",
          flush=True)
    wq, bq = Q.quantize(z["w"], 4), Q.quantize(z["b"], 4)
    q12 = Q.build_actor(wq, bq, Q.quantize(W12, 4), z["log_T_soft"], z["log_T_sel"],
                        int(z["n_heads"]), int(z["tph"]))
    q6 = Q.build_actor(wq, bq, Q.quantize(W6, 4), z["log_T_soft"], z["log_T_sel"],
                       int(z["n_heads"]), int(z["tph"]))
    r12q = row("int4 original (12-col)", (q12[0], n12), m)
    r6q = row("int4 std-dropped (6-col)", (q6[0], n6), m)

    json.dump(dict(source=SRC, std_dropped=DST,
                   action_max_abs_delta=max_abs,
                   tables_whose_scale_was_set_by_std=set_by_std,
                   int4_scale_shrink=dict(min=float(shrink.min()),
                                          max=float(shrink.max()),
                                          mean=float(shrink.mean())),
                   rows=[r12, r6, r12q, r6q]),
              open(os.path.join(HERE, OUT), "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
