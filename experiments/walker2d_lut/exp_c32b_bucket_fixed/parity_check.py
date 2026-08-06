"""exp_c32 — assert the JAX BucketLIF port matches the torch reference, value and gradient.

Half two of the parity test; runs in the MJX venv (jax, no torch).

WHAT IS CHECKED, beyond the usual forward/gradient pairing:

  A. hard / soft / st forwards, and `st == hard` on the JAX side.
  B. THE INTERMEDIATES, not just the outputs. `t_hard`, `t_soft`, the soft partition `g`
     and the `boundaries` themselves are all dumped and compared. Bucket addressing can
     be wrong in a way the final output hides: if the boundary cumsum ran along the wrong
     axis, most samples would still land in *some* bucket and the output would merely be
     wrong rather than malformed. Comparing the boundaries directly catches it.
  C. gradient parity on EVERY parameter, including `beta_raw` and `beta_base` -- the two
     that only reach the loss through the soft partition.
  D. THE PARTITION SUMS TO ONE. sum_m g_m must be 1 for every (sample, table). A partition
     that summed to 0.97 would train perfectly happily and silently scale every addressed
     row; nothing else in this harness would notice.
  E. THE BOUNDARIES ARE STRICTLY INCREASING. This is the invariant the softplus-cumsum
     parameterisation exists to guarantee; asserting it here means a later refactor that
     breaks it fails loudly.
  F. the table gradient is a hard one-row-per-table scatter, and no parameter is dead.

Tolerances are relative-to-scale, as in exp_c30/c31.

Usage:
  python parity_check.py REF.npz
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

import jax_bucket_lif as M

TOL = 2e-5
CASES = ("init", "perturbed")


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def main():
    z = np.load(sys.argv[1])
    tph, heads = int(z["tables_per_head"]), int(z["n_heads"])
    nb, n_out = int(z["n_buckets"]), int(z["n_outputs"])
    eps = float(z["eps"])
    n_tables = heads * tph
    fails = []

    def check(name, got, ref):
        r = rel(got, ref)
        ok = r < TOL
        fails.append(None if ok else name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<38} rel {r:.3e}")

    def check_bool(name, ok, detail):
        fails.append(None if ok else name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<38} {detail}")

    print(f"parity at input_dim={int(z['input_dim'])} heads={heads} tph={tph} "
          f"n_buckets={nb} n_out={n_out}  (torch reports {int(z['n_params']):,} params)")

    for case in CASES:
        print(f"\n--- case: {case} ---")
        p = {k[len(f'p_{case}_'):]: jnp.asarray(z[k])
             for k in z.files if k.startswith(f"p_{case}_")}
        x = jnp.asarray(z[f"x_{case}"])
        gout = jnp.asarray(z[f"gout_{case}"])

        # --- B: the intermediates ------------------------------------------
        bnd = M.boundaries(p)
        check(f"{case}: boundaries", bnd, z[f"bnd_{case}"])
        t_hard, t_soft = M.first_spike(p, x, n_tables)
        check(f"{case}: t_hard", t_hard, z[f"t_hard_{case}"])
        check(f"{case}: t_soft", t_soft, z[f"t_soft_{case}"])
        g_soft = M.bucket_soft(p, t_soft, n_tables)
        check(f"{case}: soft partition g", g_soft, z[f"g_soft_{case}"])

        # --- A: forwards ----------------------------------------------------
        y_hard = M.apply(p, x, eps, heads, tph, nb, mode="hard")
        y_soft = M.apply(p, x, eps, heads, tph, nb, mode="soft")
        y_st = M.apply(p, x, eps, heads, tph, nb, mode="st")
        check(f"{case}: forward hard", y_hard, z[f"y_hard_{case}"])
        check(f"{case}: forward soft", y_soft, z[f"y_soft_{case}"])
        check(f"{case}: forward st", y_st, z[f"y_st_{case}"])
        d = float(np.abs(np.asarray(y_st) - np.asarray(y_hard)).max())
        check_bool(f"{case}: st == hard (jax side)", d < 1e-6, f"max|diff| {d:.3e}")

        # --- D/E: the two structural invariants -----------------------------
        s = float(np.abs(np.asarray(g_soft).sum(-1) - 1.0).max())
        check_bool(f"{case}: partition sums to 1", s < 1e-5, f"max|Σg − 1| {s:.3e}")
        inc = bool(np.all(np.asarray(bnd)[:, 1:] > np.asarray(bnd)[:, :-1]))
        check_bool(f"{case}: boundaries strictly increasing", inc,
                   f"min gap {float(np.diff(np.asarray(bnd), axis=-1).min()):.4f}")

        # --- C: gradient parity on every parameter --------------------------
        def loss(pp):
            return (M.apply(pp, x, eps, heads, tph, nb, mode="st") * gout).sum()

        g = jax.grad(loss)(p)
        for k in sorted(g):
            check(f"{case}: grad {k}", g[k], z[f"g_{case}_{k}"])

        # --- F: decoupling is real; nothing dead ----------------------------
        nz = int((np.abs(np.asarray(g["table"])).sum(-1) > 0).sum())
        check_bool(f"{case}: table grad is a scatter", nz <= x.shape[0] * n_tables,
                   f"{nz} of {n_tables*nb} rows nonzero, cap {x.shape[0]*n_tables}")
        dead = [k for k in sorted(g) if float(np.abs(np.asarray(g[k])).max()) == 0.0]
        check_bool(f"{case}: no dead parameter", not dead,
                   "all 8 receive gradient" if not dead else f"DEAD: {dead}")

    bad = [f for f in fails if f]
    if bad:
        print(f"\nPARITY FAILED on {len(bad)}: {', '.join(bad)}")
        sys.exit(1)
    print(f"\nPARITY OK — {len(fails)} checks over {len(CASES)} cases, "
          f"all within {TOL:.0e} relative")


if __name__ == "__main__":
    main()
