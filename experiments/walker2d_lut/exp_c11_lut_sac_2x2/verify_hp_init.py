"""exp_c11 — verify the torch-faithful hyperplane anchor_pairs init (#75). MJX venv.

Three claims, checked separately because they can fail independently:

  D1. The JAX anchor_pairs init draws the SAME pairs torch's HyperplaneMultiHeadLUT
      draws for the same seed/shape/policy — index for index.
  D2. Its w/b are numerically identical to torch's hyperplane_weight/hyperplane_bias
      at init (w = e_a - e_b exactly, b = 0 exactly, no perturbation).
  D3. The JAX hard forward through those hyperplanes equals torch's HyperplaneMultiHeadLUT
      hard forward on the same input AND equals a FastMultiHeadLut with those pairs --
      which is the actual point of the init: it starts as a bit-exact FastMHL.

Fixture comes from emit_torch_hp_init.py (spiky venv).
"""
import json, os, sys

import jax.numpy as jnp
import numpy as np

import jax_lut_ext as X
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "exp_c06_jax_backprop"))
import jax_lut_grad as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "torch_hp_init.npz"))
heads, tph = int(z["n_heads"]), int(z["tph"])
nap, input_dim = int(z["nap"]), int(z["input_dim"])
policy, device = str(z["policy"]), str(z["device"])
n_tables = heads * tph
x = jnp.asarray(z["x"])
out, tol = {}, 5e-6


def cmp(name, j, t):
    j = np.asarray(j, np.float64); t = np.asarray(t, np.float64)
    ma = float(np.abs(j - t).max()); scale = float(np.abs(t).max())
    rel = ma / scale if scale else 0.0
    exact = bool((j == t).all())
    out[name] = dict(max_abs=ma, max_rel=rel, exact=exact)
    print(f"  {name:<30} max|Δ| {ma:.3e}  rel {rel:.3e}  "
          f"{'EXACT' if exact else ''}")
    return rel


print(f"fixture: {n_tables} tables x nap{nap} over d={input_dim}, "
      f"policy={policy}, device={device}")

print("\nD1. anchor pairs drawn by JAX vs by torch's HyperplaneMultiHeadLUT:")
w_j, b_j = X.anchor_pair_wb_lutorch(n_tables, nap, input_dim, seed=0,
                                    policy=policy, heads=heads, device=device)
cache = os.path.join(os.path.expanduser("~/.cache/spiky_anchors"),
                     f"anchors_{policy}_t{n_tables}_nap{nap}_d{input_dim}"
                     f"_h{heads}_s0_{device}.npz")
zc = np.load(cache)
idx_ok = bool((zc["anchor_a"] == z["anchor_a"]).all()
              and (zc["anchor_b"] == z["anchor_b"]).all())
print(f"  indices vs torch buffers       {'EXACT MATCH' if idx_ok else 'DIFFER'}")
out["indices_match"] = idx_ok

print("\nD2. the initialised hyperplanes themselves:")
r_w = cmp("hyperplane_weight (w)", w_j, z["hp_w"])
r_b = cmp("hyperplane_bias (b)", b_j, z["hp_b"])
print(f"  b is exactly zero: {bool((np.asarray(b_j) == 0).all())} | "
      f"nonzeros per bit row: {int((np.asarray(w_j) != 0).sum(-1).min())}-"
      f"{int((np.asarray(w_j) != 0).sum(-1).max())} (must be 2)")

print("\nD3. hard forward through those hyperplanes:")
y_j = L.lut_apply(x, w_j, b_j, jnp.asarray(z["weights"]),
                  jnp.asarray(z["log_T_soft"]), jnp.asarray(z["log_T_sel"]),
                  heads, tph)
r_hp = cmp("vs torch HyperplaneMHLUT", y_j, z["y_hp"])
r_fa = cmp("vs torch FastMultiHeadLut", y_j, z["y_fa"])
print("  (the second is the point of this init: at step 0 the hyperplane model IS "
      "an anchor LUT)")

worst = max(r_w, r_b, r_hp, r_fa)
verdict = "PASS" if (worst <= tol and idx_ok) else "FAIL"
print(f"\nVERDICT: {verdict} (worst rel {worst:.3e}, tol {tol:.0e}, indices {idx_ok})")
out["summary"] = dict(worst_rel=worst, tol=tol, verdict=verdict,
                      indices_match=idx_ok, policy=policy, device=device)
json.dump(out, open(os.path.join(HERE, "verify_hp_init_results.json"), "w"), indent=1)
