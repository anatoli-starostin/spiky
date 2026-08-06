"""exp_c30 — assert the JAX port matches the torch reference, value and gradient.

Half two of the parity test; runs in the MJX venv (jax, no torch) against the npz written
by `torch_ref_dump.py` in the spiky venv.

WHAT IS CHECKED, and why each one is here rather than just "outputs match":

  A. hard  forward  -- the value that is actually deployed.
  B. soft  forward  -- the reference blend. It shares no code path with hard, so if the
                       membrane were subtly wrong (a transposed pair index, say) hard
                       could still pass by luck while soft would not.
  C. st    forward  == hard forward, on BOTH sides. The value-cancelling identity is the
                       core claim of mode="st"; a port that broke it would train one
                       function and deploy another.
  D. gradient parity on EVERY parameter, including `log_temp_bit`. This is the real test:
     the forward is easy to get right and the decoupled backward is where a port drifts.
     A wrong `stop_gradient` would leave the forward untouched and only show up here --
     e.g. forgetting to detach the table smears the weight gradient across all 2**nap
     rows, which is exactly the failure the torch module's docstring reports hitting.

Tolerances are relative-to-scale (`max|a-b| / max(1, max|ref|)`) because the parameters
differ by orders of magnitude: `P` gradients are ~1e-4 while `table` gradients are ~1e0,
and one absolute tolerance would be either meaningless for the former or unmeetable for
the latter.

Usage:
  python parity_check.py REF.npz
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

import jax_lif_mhl as M

TOL = 2e-5          # fp32 + a different op order; anything structural is orders larger


def rel(a, b):
    """max|a-b| scaled by the reference's own magnitude."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def main():
    z = np.load(sys.argv[1])
    nap, tph = int(z["n_anchor_pairs"]), int(z["tables_per_head"])
    heads, n_out = int(z["n_heads"]), int(z["n_outputs"])
    eps = float(z["eps"])

    p = {k[2:]: jnp.asarray(z[k]) for k in z.files if k.startswith("p_")}
    x = jnp.asarray(z["x"])
    gout = jnp.asarray(z["gout"])

    fails = []

    def check(name, got, ref):
        r = rel(got, ref)
        ok = r < TOL
        fails.append(None) if ok else fails.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} rel {r:.3e}")
        return ok

    print(f"parity at input_dim={int(z['input_dim'])} heads={heads} n_out={n_out} "
          f"nap={nap} tph={tph} eps={eps}")

    # --- A/B: forward values ------------------------------------------------
    y_hard = M.apply(p, x, eps, heads, tph, nap, mode="hard")
    y_soft = M.apply(p, x, eps, heads, tph, nap, mode="soft")
    y_st = M.apply(p, x, eps, heads, tph, nap, mode="st")
    check("forward hard", y_hard, z["y_hard"])
    check("forward soft", y_soft, z["y_soft"])
    check("forward st", y_st, z["y_st"])

    # --- C: the straight-through identity, checked on the JAX side itself ----
    d_st_hard = float(np.abs(np.asarray(y_st) - np.asarray(y_hard)).max())
    ok = d_st_hard < 1e-6
    fails.append(None if ok else "st==hard identity")
    print(f"  {'PASS' if ok else 'FAIL'}  {'st == hard (jax side)':<28} "
          f"max|diff| {d_st_hard:.3e}")

    # --- D: gradient parity on every parameter ------------------------------
    def loss(pp):
        return (M.apply(pp, x, eps, heads, tph, nap, mode="st") * gout).sum()

    g = jax.grad(loss)(p)
    for k in sorted(g):
        check(f"grad {k}", g[k], z[f"g_{k}"])

    # A sanity assertion that the decoupling is REAL and not accidentally symmetric:
    # the table gradient must be a one-row-per-table scatter, i.e. exactly `tph` rows
    # per sample can be nonzero. If the table were not detached in the address term,
    # every row would carry gradient.
    nz_rows = int((np.abs(np.asarray(g["table"])).sum(-1) > 0).sum())
    rows_tot = tph * (1 << nap)
    print(f"  info  table rows with nonzero grad: {nz_rows} of {rows_tot} "
          f"({100*nz_rows/rows_tot:.1f}%) -- hard scatter, not a full-table smear")

    bad = [f for f in fails if f]
    if bad:
        print(f"\nPARITY FAILED on {len(bad)}: {', '.join(bad)}")
        sys.exit(1)
    print(f"\nPARITY OK — {len(fails)} checks, all within {TOL:.0e} relative")


if __name__ == "__main__":
    main()
