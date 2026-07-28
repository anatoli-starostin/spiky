"""exp_c11 — verify the two JAX extensions against torch (#75). MJX venv.

  A. hybrid_smooth forward  vs  HyperplaneMultiHeadLUT(forward_mode="hybrid_smooth")
  B. anchor-pair-as-hyperplane  vs  FastMultiHeadLut(forward_mode="hard")

(B) is the sharper test: it checks that writing w = e_a - e_b, b = 0 into the general
affine front-end really does reproduce anchor-pair addressing, which is the whole basis
of the "no new forward needed" claim.
"""
import json, os

import jax, jax.numpy as jnp
import numpy as np

import jax_lut_ext as X
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "exp_c06_jax_backprop"))
import jax_lut_grad as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "torch_ext.npz"))
heads, tph = int(z["n_heads"]), int(z["tph"])
x = jnp.asarray(z["x"])
out = {}


def cmp(name, j, t):
    j = np.asarray(j, np.float64); t = np.asarray(t, np.float64)
    ma = float(np.abs(j - t).max()); scale = float(np.abs(t).max())
    rel = ma / scale if scale else 0.0
    exact = bool((j == t).all())
    out[name] = dict(max_abs=ma, max_rel=rel, exact=exact)
    print(f"  {name:<26} max|Δ| {ma:.3e}  rel {rel:.3e}  "
          f"{'EXACT' if exact else ''}")
    return rel


print("A. hybrid_smooth forward (JAX vs torch):")
ys = X.lut_apply_smooth(x, jnp.asarray(z["w"]), jnp.asarray(z["b"]),
                        jnp.asarray(z["weights"]), jnp.asarray(z["log_T_soft"]),
                        jnp.asarray(z["log_T_sel"]), heads, tph)
r_smooth = cmp("y_hybrid_smooth", ys, z["y_smooth"])

print("B. anchor pairs written as a frozen hyperplane (JAX vs torch FastMHL, hard):")
a_idx, b_idx = z["anchor_a"], z["anchor_b"]
n_tables, nap = a_idx.shape
w = np.zeros((n_tables, nap, x.shape[1]), np.float32)
for t in range(n_tables):
    for i in range(nap):
        w[t, i, a_idx[t, i]] = 1.0        # bit = 1[x[a] - x[b] > 0]
        w[t, i, b_idx[t, i]] = -1.0
ya = L.lut_apply(x, jnp.asarray(w), jnp.zeros((n_tables, nap), jnp.float32),
                 jnp.asarray(z["fa_weights"]), jnp.asarray(z["log_T_soft"]),
                 jnp.asarray(z["log_T_sel"]), heads, tph)
r_anchor = cmp("y_anchor_pairs", ya, z["y_anchor"])

tol = 5e-6
verdict = "PASS" if max(r_smooth, r_anchor) <= tol else "FAIL"
print(f"\nVERDICT: {verdict} (worst rel {max(r_smooth, r_anchor):.3e}, tol {tol:.0e})")
out["summary"] = dict(worst_rel=max(r_smooth, r_anchor), tol=tol, verdict=verdict)
json.dump(out, open(os.path.join(HERE, "verify_ext_results.json"), "w"), indent=1)
raise SystemExit(0 if verdict == "PASS" else 1)
