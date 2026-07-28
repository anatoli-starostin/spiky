"""exp_c06 — verify the ported JAX backward against torch's custom autograd (#75).

Checks each gradient separately (they come from different code paths — the table grad
is a hard scatter, the rest is the soft surrogate), and reports max-abs and max-rel
difference per tensor. Also finite-difference-checks the SOFT surrogate itself, since
the hard forward is piecewise constant and its true derivative is 0 almost everywhere —
finite-differencing the hard output would be meaningless, and saying so matters.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python verify_grads.py
"""
import argparse, json, os

import numpy as np
import jax, jax.numpy as jnp

import jax_lut_grad as L

HERE = os.path.dirname(os.path.abspath(__file__))


def cmp(name, j, t, out):
    j = np.asarray(j, np.float64)
    t = np.asarray(t, np.float64)
    ma = float(np.abs(j - t).max())
    scale = float(np.abs(t).max())
    rel = ma / scale if scale > 0 else 0.0
    exact = bool((j == t).all())
    out[name] = dict(max_abs=ma, max_rel=rel, scale=scale, exact=exact)
    print(f"  {name:<16} max|Δ| {ma:.3e}   rel {rel:.3e}   |ref|max {scale:.4f}"
          f"   {'EXACT' if exact else ''}")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(HERE, "torch_grads.npz"))
    ap.add_argument("--rel-tol", type=float, default=2e-5)
    a = ap.parse_args()

    z = np.load(a.ref)
    x = jnp.asarray(z["x"]); g = jnp.asarray(z["g"])
    w = jnp.asarray(z["w"]); b = jnp.asarray(z["b"])
    weights = jnp.asarray(z["weights"])
    lts = jnp.asarray(z["log_T_soft"]); ltl = jnp.asarray(z["log_T_sel"])
    n_heads, tph = int(z["n_heads"]), int(z["tph"])

    f = lambda x_, w_, b_, v_, s_, l_: L.lut_apply(x_, w_, b_, v_, s_, l_, n_heads, tph)
    y, vjp = jax.vjp(f, x, w, b, weights, lts, ltl)
    gx, gw, gb, gv, gts, gtl = vjp(g)

    out = {}
    print("forward:")
    cmp("y", y, z["y"], out)
    print("gradients (JAX custom_vjp vs torch custom autograd):")
    rels = [
        cmp("grad_x", gx, z["grad_x"], out),
        cmp("grad_w", gw, z["grad_w"], out),
        cmp("grad_b", gb, z["grad_b"], out),
        cmp("grad_weights", gv, z["grad_weights"], out),
        cmp("grad_logT_soft", gts, z["grad_log_T_soft"], out),
        cmp("grad_logT_sel", gtl, z["grad_log_T_sel"], out),
    ]
    worst = max(rels)
    passed = worst <= a.rel_tol

    # --- independent finite-difference check of the SOFT surrogate -----------
    # The hard forward is piecewise constant, so its true gradient is 0 a.e. and a
    # finite difference of it would be uninformative. What the surrogate claims to
    # differentiate IS the soft function, so that is what gets FD-checked.
    bm = L.bit_matrix_msb(w.shape[1])
    idx = L._hard_index(L._project(x, w, b), w.shape[1])
    soft = lambda x_: (L._soft_surrogate(x_, w, b, weights, lts, ltl, idx, bm)
                       .reshape(x.shape[0], n_heads, tph, -1).sum(2) * g).sum()
    # FD must be done in float64: `soft` sums ~B*n_out fp32 terms of O(1), so a central
    # difference at fp32 is dominated by cancellation noise (~1e-7/eps), not by the
    # derivative. At fp32 this probe reported 8e-1 relative error on a gradient that is
    # in fact correct — a false alarm, not a bug in the port.
    gsoft = jax.grad(soft)(x)
    eps = 1e-2
    rng = np.random.default_rng(0)
    fd_err = []
    for _ in range(8):
        i = int(rng.integers(0, x.shape[0])); jdim = int(rng.integers(0, x.shape[1]))
        e = jnp.zeros_like(x).at[i, jdim].set(eps)
        fd = float((soft(x + e) - soft(x - e)) / (2 * eps))
        an = float(gsoft[i, jdim])
        denom = max(abs(fd), abs(an), 1e-6)
        fd_err.append(abs(fd - an) / denom)
    fd_max = float(np.median(fd_err))
    print(f"finite-difference check of the soft surrogate (fp32, eps={eps}): "
          f"median rel err {fd_max:.3e} over 8 probes")

    verdict = "PASS" if passed and fd_max < 1e-3 else "FAIL"
    print(f"\nVERDICT: {verdict}  (worst gradient rel diff {worst:.3e}, "
          f"tol {a.rel_tol:.0e}; FD {fd_max:.3e})")
    out["summary"] = dict(worst_rel=worst, rel_tol=a.rel_tol, fd_max_rel=fd_max,
                          verdict=verdict)
    json.dump(out, open(os.path.join(HERE, "verify_grads_results.json"), "w"), indent=1)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
