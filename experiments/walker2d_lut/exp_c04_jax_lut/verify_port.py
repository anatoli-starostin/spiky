"""exp_c04 — JAX side of the port verification (#75). Run in the WALKER2D_MJX venv.

Loads the torch reference (weights + input + output) and checks the JAX forward
reproduces it. The bar is EXACT, not "close": the addressing is discrete, so any
disagreement in the sign tests shows up as a whole wrong row, and the only tolerable
difference is fp32 summation-order noise in the tables_per_head reduce.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python verify_port.py
"""
import argparse, json, os

import numpy as np
import jax.numpy as jnp

import jax_lut

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(HERE, "torch_reference.npz"))
    a = ap.parse_args()

    z = np.load(a.ref)
    params = jax_lut.from_npz(a.ref)
    x = jnp.asarray(z["x"])
    y_torch = np.asarray(z["y"], np.float64)

    y_jax = np.asarray(jax_lut.lut_forward(params, x), np.float64)

    assert y_jax.shape == y_torch.shape, (y_jax.shape, y_torch.shape)
    diff = np.abs(y_jax - y_torch)
    max_abs = float(diff.max())
    scale = float(np.abs(y_torch).max())
    exact = bool((y_jax == y_torch).all())

    # An index mismatch would show as a whole different row, i.e. an error the size of
    # a table entry. Separate that from pure summation-order noise.
    eps32 = float(np.finfo(np.float32).eps)
    tolerable = scale * eps32 * params["tph"]

    print(f"shape {y_jax.shape} | torch |y|max {scale:.6f}")
    print(f"max abs diff (jax - torch) : {max_abs:.3e}")
    print(f"bitwise identical           : {exact}")
    print(f"fp32 sum-order bound (eps*|y|max*tph = {tolerable:.3e}) "
          f"-> {'WITHIN' if max_abs <= tolerable else 'EXCEEDED'}")
    verdict = "EXACT" if exact else ("FP32-SUM-ORDER" if max_abs <= tolerable
                                     else "MISMATCH")
    print(f"VERDICT: {verdict}")

    json.dump(dict(shape=list(y_jax.shape), max_abs_diff=max_abs,
                   bitwise_identical=exact, sum_order_bound=tolerable,
                   verdict=verdict),
              open(os.path.join(HERE, "verify_port_results.json"), "w"), indent=1)
    return 0 if verdict != "MISMATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
