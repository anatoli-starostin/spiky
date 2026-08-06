"""exp_c31 — assert the JAX PureLIF port matches the torch reference, value and gradient.

Half two of the parity test; runs in the MJX venv (jax, no torch) against the npz written
by `torch_ref_dump.py` in the spiky venv.

WHAT IS CHECKED, and why each one is here rather than just "outputs match":

  A. hard forward -- the value that is actually deployed.
  B. soft forward -- the reference blend. It shares the membrane with hard but reaches the
     bits by a completely different route (the smooth first-success over sorted arrivals
     rather than an argmax), so a transposed `dt` or a mis-shifted survival product can
     leave hard correct and soft wrong.
  C. st forward == hard forward, on BOTH sides. The value-cancelling identity is the core
     claim of mode="st"; a port that broke it would train one function and deploy another.
  D. gradient parity on EVERY parameter. This is the real test: the forward is easy to get
     right and the decoupled backward is where a port drifts. A wrong `stop_gradient`
     leaves the forward untouched and only shows up here.
  E. the table gradient is a HARD SCATTER -- at most `tph` rows per sample can be nonzero.
     If the table were not detached in the address term, every one of the 2**nap rows
     would carry gradient. This is the failure the reference module's own docstring
     reports hitting.
  F. no parameter is DEAD. Every one of the seven tensors must receive nonzero gradient;
     `L`, `tau_raw` and `log_T_cross` all reach the loss only through the soft path, so a
     missing term there would show as an exactly-zero gradient rather than a wrong one.

Run over BOTH dumped cases. `init` alone is not sufficient: at init every per-table
parameter is identical and `delay` is zero, so the (n_tables, nap) grouping is invisible.
`perturbed` breaks that symmetry.

Tolerances are relative-to-scale (`max|a-b| / max(1, max|ref|)`) because the parameters
differ by orders of magnitude: `tau_raw` gradients are ~1e1 while `log_temp_bit` gradients
are ~1e-2, and one absolute tolerance would be either meaningless for the former or
unmeetable for the latter.

Usage:
  python parity_check.py REF.npz
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

import jax_pure_lif as M

TOL = 2e-5          # fp32 + a different op order; anything structural is orders larger
CASES = ("init", "perturbed")


def rel(a, b):
    """max|a-b| scaled by the reference's own magnitude."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def main():
    z = np.load(sys.argv[1])
    nap, tph = int(z["n_anchor_pairs"]), int(z["tables_per_head"])
    heads, n_out = int(z["n_heads"]), int(z["n_outputs"])
    eps = float(z["eps"])
    fails = []

    def check(name, got, ref):
        r = rel(got, ref)
        ok = r < TOL
        fails.append(None if ok else name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} rel {r:.3e}")

    def check_bool(name, ok, detail):
        fails.append(None if ok else name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} {detail}")

    print(f"parity at input_dim={int(z['input_dim'])} heads={heads} n_out={n_out} "
          f"nap={nap} tph={tph}  (torch reports {int(z['n_params']):,} params)")

    for case in CASES:
        print(f"\n--- case: {case} ---")
        p = {k[len(f'p_{case}_'):]: jnp.asarray(z[k])
             for k in z.files if k.startswith(f"p_{case}_")}
        x = jnp.asarray(z[f"x_{case}"])
        gout = jnp.asarray(z[f"gout_{case}"])

        # --- A/B: forward values --------------------------------------------
        y_hard = M.apply(p, x, eps, heads, tph, nap, mode="hard")
        y_soft = M.apply(p, x, eps, heads, tph, nap, mode="soft")
        y_st = M.apply(p, x, eps, heads, tph, nap, mode="st")
        check(f"{case}: forward hard", y_hard, z[f"y_hard_{case}"])
        check(f"{case}: forward soft", y_soft, z[f"y_soft_{case}"])
        check(f"{case}: forward st", y_st, z[f"y_st_{case}"])

        # --- C: the straight-through identity, on the JAX side itself --------
        d = float(np.abs(np.asarray(y_st) - np.asarray(y_hard)).max())
        check_bool(f"{case}: st == hard (jax side)", d < 1e-6, f"max|diff| {d:.3e}")

        # --- D: gradient parity on every parameter ---------------------------
        def loss(pp):
            return (M.apply(pp, x, eps, heads, tph, nap, mode="st") * gout).sum()

        g = jax.grad(loss)(p)
        for k in sorted(g):
            check(f"{case}: grad {k}", g[k], z[f"g_{case}_{k}"])

        # --- E: the decoupling is real, not accidentally symmetric ------------
        nz_rows = int((np.abs(np.asarray(g["table"])).sum(-1) > 0).sum())
        rows_tot = tph * (1 << nap)
        check_bool(f"{case}: table grad is a scatter", nz_rows <= x.shape[0] * tph,
                   f"{nz_rows} of {rows_tot} rows nonzero "
                   f"({100*nz_rows/rows_tot:.1f}%), cap {x.shape[0]*tph}")

        # --- F: nothing dead --------------------------------------------------
        dead = [k for k in sorted(g) if float(np.abs(np.asarray(g[k])).max()) == 0.0]
        check_bool(f"{case}: no dead parameter", not dead,
                   "all 7 receive gradient" if not dead else f"DEAD: {dead}")

    bad = [f for f in fails if f]
    if bad:
        print(f"\nPARITY FAILED on {len(bad)}: {', '.join(bad)}")
        sys.exit(1)
    print(f"\nPARITY OK — {len(fails)} checks over {len(CASES)} cases, "
          f"all within {TOL:.0e} relative")


if __name__ == "__main__":
    main()
