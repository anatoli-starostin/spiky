"""exp_c11 — verify the JAX hybrid_smooth FORWARD and BACKWARD against torch (#75).

Written after a real bug: the smooth backward originally reused the hard-mode backward,
whose table-weight gradient is a SINGLE-row scatter at `main`. The smooth forward blends
two rows, so its weight gradient must be a TWO-row scatter — (1-u) at `main`, u at `alt`.
Everything else (x, hyperplanes, temperatures) is shared between the modes, which is why
only one param group was wrong and why it still trained.

This checks every param group separately so a repeat of that mistake cannot hide behind
an aggregate norm.
"""
import argparse, json, os

import jax, jax.numpy as jnp
import numpy as np

import jax_lut_ext as X

HERE = os.path.dirname(os.path.abspath(__file__))


def cmp(name, j, t, out):
    j = np.asarray(j, np.float64); t = np.asarray(t, np.float64)
    ma = float(np.abs(j - t).max()); scale = float(np.abs(t).max())
    rel = ma / scale if scale else 0.0
    out[name] = dict(max_abs=ma, max_rel=rel, ref_scale=scale)
    print(f"  {name:<18} max|Δ| {ma:.3e}   rel {rel:.3e}   |ref|max {scale:.4f}")
    return rel


def check(npz, tol, out, label):
    z = np.load(os.path.join(HERE, npz))
    heads, tph = int(z["n_heads"]), int(z["tph"])
    x, g = jnp.asarray(z["x"]), jnp.asarray(z["g"])
    args = (x, jnp.asarray(z["w"]), jnp.asarray(z["b"]), jnp.asarray(z["weights"]),
            jnp.asarray(z["log_T_soft"]), jnp.asarray(z["log_T_sel"]))
    f = lambda *p: X.lut_apply_smooth(*p, heads, tph)
    y, vjp = jax.vjp(f, *args)
    gx, gw, gb, gv, gts, gtl = vjp(g)

    print(f"\n[{label}]")
    o = {}
    rels = [cmp("y (forward)", y, z["y"], o),
            cmp("grad_x", gx, z["grad_x"], o),
            cmp("grad_w", gw, z["grad_w"], o),
            cmp("grad_b", gb, z["grad_b"], o),
            cmp("grad_weights", gv, z["grad_weights"], o),
            cmp("grad_logT_soft", gts, z["grad_log_T_soft"], o),
            cmp("grad_logT_sel", gtl, z["grad_log_T_sel"], o)]
    worst = max(rels)
    o["worst_rel"] = worst
    o["verdict"] = "PASS" if worst <= tol else "FAIL"
    print(f"  -> worst rel {worst:.3e}  {o['verdict']}")
    out[label] = o
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5e-5)
    a = ap.parse_args()
    out = {}
    worst = [check("torch_smooth_grads.npz", a.tol, out, "nominal")]
    for extra, label in (("torch_smooth_grads_hot.npz", "extreme T (soft=4, sel=4)"),
                         ("torch_smooth_grads_cold.npz", "extreme T (soft=.05, sel=.05)")):
        if os.path.exists(os.path.join(HERE, extra)):
            worst.append(check(extra, a.tol, out, label))
    w = max(worst)
    verdict = "PASS" if w <= a.tol else "FAIL"
    print(f"\nOVERALL: {verdict} (worst rel {w:.3e}, tol {a.tol:.0e})")
    out["summary"] = dict(worst_rel=w, tol=a.tol, verdict=verdict)
    json.dump(out, open(os.path.join(HERE, "verify_smooth_grads_results.json"), "w"),
              indent=1)
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
