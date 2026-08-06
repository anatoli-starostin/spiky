"""exp_c32b — our recurrence vs nucstar's cumsum factorisation: agreement and safety margin.

Both compute the same quantity, V_k = sum_{j<=k} w_j exp(-(a_k - a_j)/tau), exactly:

  ours (jax_bucket_lif.membrane_linear)
      V_k = w_k + exp(-(a_k - a_{k-1})/tau) * V_{k-1}
      every intermediate is O(V), every decay factor is in (0, 1].

  nucstar's (bucket_lif_detectors_mhl._first_spike @ 0024b81f)
      V_k = exp(-a_k/tau) * cumsum_{j<=k}( w_j * exp(a_j/tau) )
      the intermediate cumsum reaches exp(t_window/tau), then is rescaled back.

They are algebraically identical. They are NOT identically conditioned, and his own comment
says so ("Do NOT lower the tau floor -- that is what keeps this factorization stable"). This
script measures two things:

  1. AGREEMENT at the configuration we actually run (tau >= 1.0, t_window = 32).
  2. THE SAFETY MARGIN: how far tau can fall, or t_window rise, before the cumsum form
     loses precision and then overflows in float32. The recurrence form is used as ground
     truth in float64.

Reported so the floor can be justified with a number rather than an assertion.

Usage:
  python compare_membranes.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax_bucket_lif as M                                  # noqa: E402

N, T, B = 17, 32, 64


def cumsum_form(a_srt, w_srt, tau, dtype):
    """nucstar's factorisation, in the requested precision."""
    a, w, tt = (x.astype(dtype) for x in (a_srt, w_srt, tau))
    return jnp.exp(-a / tt) * jnp.cumsum(w * jnp.exp(a / tt), axis=-1)


def recurrence_form(a_srt, w_srt, tau, dtype):
    a, w, tt = (x.astype(dtype) for x in (a_srt, w_srt, tau))
    return M.membrane_linear(a, w, tt)


def sample(key, t_window):
    ka, kw = jax.random.split(key)
    a = jnp.sort(jax.random.uniform(ka, (B, T, N), minval=0.0, maxval=t_window), axis=-1)
    w = 2.0 * jax.nn.sigmoid(-2.2 + 0.5 * jax.random.normal(kw, (B, T, N)))
    return a, w


def rel(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    d = np.abs(x - ref) / np.maximum(np.abs(ref), 1e-30)
    return float(np.nanmax(d)), float(np.nanmedian(d))


def main():
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)

    print("=== 1. Agreement at the configuration we run (t_window=32, tau>=1.0) ===")
    a, w = sample(key, 32.0)
    for tv in (1.0, 2.313, 5.0, 10.0):
        tau = jnp.full((1, T, 1), tv)
        c32 = cumsum_form(a, w, tau, jnp.float32)
        r32 = recurrence_form(a, w, tau, jnp.float32)
        r64 = recurrence_form(a, w, tau, jnp.float64)
        mc, _ = rel(c32, r64)
        mr, _ = rel(r32, r64)
        print(f"  tau={tv:6.3f}   cumsum vs fp64 {mc:9.2e}   "
              f"recurrence vs fp64 {mr:9.2e}   cumsum/recurrence error "
              f"{mc/max(mr,1e-30):6.1f}x")

    print("\n=== 2. Safety margin: lowering tau below the 1.0 floor (t_window=32) ===")
    print("  (the floor exists to prevent this; measuring what it prevents)")
    for tv in (1.0, 0.7, 0.5, 0.4, 0.36, 0.3, 0.2, 0.1):
        tau = jnp.full((1, T, 1), tv)
        c32 = cumsum_form(a, w, tau, jnp.float32)
        r64 = recurrence_form(a, w, tau, jnp.float64)
        bad = bool(np.any(~np.isfinite(np.asarray(c32))))
        mc, _ = rel(c32, r64)
        flag = "  <-- OVERFLOW (inf/nan)" if bad else ("  <-- >1% error" if mc > 1e-2
                                                       else "")
        print(f"  tau={tv:5.2f}   exp(t_window/tau)={np.exp(32.0/tv):9.2e}   "
              f"cumsum max rel err {mc:9.2e}{flag}")

    print("\n=== 3. Safety margin: raising t_window at the tau=1.0 floor ===")
    for tw in (32.0, 60.0, 80.0, 88.0, 100.0, 120.0):
        aa, ww = sample(key, tw)
        tau = jnp.full((1, T, 1), 1.0)
        c32 = cumsum_form(aa, ww, tau, jnp.float32)
        r64 = recurrence_form(aa, ww, tau, jnp.float64)
        bad = bool(np.any(~np.isfinite(np.asarray(c32))))
        mc, _ = rel(c32, r64)
        flag = "  <-- OVERFLOW (inf/nan)" if bad else ""
        print(f"  t_window={tw:6.1f}   exp(t_window/tau)={np.exp(tw):9.2e}   "
              f"cumsum max rel err {mc:9.2e}{flag}")

    print("\n  float32 max is 3.40e+38, so exp(t_window/tau) overflows once "
          "t_window/tau > 88.7.")
    print("  At the shipped config (t_window=32, tau>=1.0) the argument is at most 32, "
          "\n  i.e. exp = 7.9e13 — a margin of 56.7 in the exponent. The floor is doing "
          "real work.")


if __name__ == "__main__":
    main()
