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

print("\nC. the SAMPLER: lutorch's own draw, reproduced through the cache:")
input_dim, nap = int(z["input_dim"]), int(z["nap"])
n_tables = heads * tph
fa_policy = str(z["anchor_policy"])

# C1 -- balanced: index-for-index against torch. Cannot be checked through a forward
# because FastMultiHeadLut REJECTS the balanced policy, so there is no torch forward
# to compare to; the indices are the whole claim.
CACHE = os.path.expanduser("~/.cache/spiky_anchors")
_, _ = X.anchor_pair_wb_lutorch(n_tables, nap, input_dim, seed=0, policy="balanced",
                                heads=heads, device="cpu")   # populates the cache
zc = np.load(os.path.join(CACHE, f"anchors_balanced_t{n_tables}_nap{nap}_"
                                 f"d{input_dim}_h{heads}_s0_cpu.npz"))
bal_ok = bool((zc["anchor_a"] == z["bal_a"]).all() and
              (zc["anchor_b"] == z["bal_b"]).all())
n_bad = int((zc["anchor_a"] != z["bal_a"]).sum() + (zc["anchor_b"] != z["bal_b"]).sum())
print(f"  balanced indices vs torch    {'EXACT MATCH' if bal_ok else f'{n_bad} DIFFER'}"
      f"  ({n_tables}x{nap} pairs)")
out["balanced_indices_match"] = dict(exact=bal_ok, n_mismatched=n_bad)

# C2 -- canonical_full_coverage: the policy FastMultiHeadLut ACTUALLY uses, so this one
# can be checked end to end -- same drawn pairs AND the same forward output.
# FastMultiHeadLut was built on cuda, so its generator was a CUDA generator: the draw
# must be reproduced on the same device or the indices differ (they did, first run).
wc, bc = X.anchor_pair_wb_lutorch(n_tables, nap, input_dim, seed=0,
                                  policy=fa_policy, heads=heads, device="cuda")
zc2 = np.load(os.path.join(CACHE, f"anchors_{fa_policy}_t{n_tables}_nap{nap}_"
                                  f"d{input_dim}_h{heads}_s0_cuda.npz"))
can_ok = bool((zc2["anchor_a"] == z["anchor_a"]).all() and
              (zc2["anchor_b"] == z["anchor_b"]).all())
print(f"  {fa_policy} indices vs FastMHL  "
      f"{'EXACT MATCH' if can_ok else 'DIFFER'}")
out["canonical_indices_match"] = dict(exact=can_ok, policy=fa_policy)
yc = L.lut_apply(x, wc, bc, jnp.asarray(z["fa_weights"]),
                 jnp.asarray(z["log_T_soft"]), jnp.asarray(z["log_T_sel"]), heads, tph)
r_sampler = cmp("y_sampler_end_to_end", yc, z["y_anchor"])

tol = 5e-6
worst = max(r_smooth, r_anchor, r_sampler)
verdict = "PASS" if (worst <= tol and bal_ok and can_ok) else "FAIL"
print(f"\nVERDICT: {verdict} (worst rel {worst:.3e}, tol {tol:.0e}; "
      f"sampler indices balanced={bal_ok} {fa_policy}={can_ok})")
out["summary"] = dict(worst_rel=worst, tol=tol, verdict=verdict,
                      balanced_indices=bal_ok, canonical_indices=can_ok)
json.dump(out, open(os.path.join(HERE, "verify_ext_results.json"), "w"), indent=1)
raise SystemExit(0 if verdict == "PASS" else 1)
