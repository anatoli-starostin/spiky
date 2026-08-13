"""exp_c53 — is the detached-hard crossing FASTER than the soft crossing?

Run in the MJX venv on the 5090, with the GPU otherwise IDLE. A contended GPU makes these
numbers meaningless, so no sweep may be running while this is trusted.

WHAT THE VARIANT REMOVES. The T_cross block in `first_spike`: a sigmoid over the full
(B,T,D,N) membrane, a cumulative-product survival along N, and a weighted sum over N. That
is work in the forward, and its VJP -- the gradient of a cumprod is itself a cumulative
construction over the same axis -- is work in the backward.

WHY THERE IS NO "backward = vgrad - forward" COLUMN, though that is the obvious way to
attribute it. It does not survive contact with the compiler. The straight-through forward is
`hard + (soft - stop_gradient(soft))`; forward-only must evaluate that expression as
written, while the differentiated trace linearises it and lets XLA drop the cancelling pair
from the primal. So forward-only is measured on a LARGER program than the primal inside
value_and_grad, and the subtraction comes out negative -- a backward costing less than
nothing. Both spellings of the benchmark produced that, which is how the artifact was found.

WHAT IS DONE INSTEAD. Two scopes, each timed forward and value_and_grad:

  first_spike   exactly the block that differs. No straight-through, no table read, so no
                cancelling pair for the compiler to fold, and the comparison is clean. This
                is where the attribution question is actually answered: if the vgrad
                speedup here far exceeds the forward speedup, the saving is in the VJP.
  apply         the whole module, as training calls it. Carries the ST fold, so its
                forward/vgrad pair is not subtractable -- but each column is still
                comparable ACROSS variants, which is the comparison that matters.

The end-to-end number that decides the practical question is neither: it is s/iter from the
training runs themselves, where exp_c50 measured ~0.22 s/iter on this exact configuration.
This microbenchmark attributes that difference; it does not replace it.

Memory is the device allocator's peak, which is what decides how many seeds fit
co-resident -- the practical constraint in this chapter, where 3-6 seeds share one card.

The flags match the training runs exactly (`--xla_gpu_deterministic_ops=true`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`); determinism changes the cost of scatter-shaped VJPs
by orders of magnitude, so benchmarking without it would measure a configuration nothing in
this chapter runs.

Usage:
  XLA_FLAGS=--xla_gpu_deterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python bench_spike_form.py
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax_mhl_lut as M          # noqa: E402

HEADS, TPH, ND, NB = 1, 128, 1, 16
BATCH, OBS = 512, 17             # the SAC update batch, so the numbers transfer
REPS, WARM = 100, 10


def timeit(fn, *args):
    for _ in range(WARM):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(REPS):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / REPS * 1e3          # ms


def peak_mb():
    try:
        st = jax.local_devices()[0].memory_stats() or {}
        return st.get("peak_bytes_in_use", 0) / 1e6
    except Exception:
        return float("nan")


def measure(form, p, x, gout):
    M.SPIKE_FORM = form

    @jax.jit
    def fs_fwd(p, x):
        t_hard, t_soft = M.first_spike(p, x)
        return t_hard + t_soft

    @jax.jit
    def fs_vg(p, x):
        return jax.value_and_grad(
            lambda pp: sum(v.sum() for v in M.first_spike(pp, x)))(p)

    @jax.jit
    def ap_fwd(p, x):
        return M.apply(p, x, HEADS, TPH, NB, ND, mode="train")

    @jax.jit
    def ap_vg(p, x, gout):
        return jax.value_and_grad(
            lambda pp: (M.apply(pp, x, HEADS, TPH, NB, ND,
                                mode="train") * gout).sum())(p)

    return dict(fs_fwd=timeit(fs_fwd, p, x), fs_vg=timeit(fs_vg, p, x),
                ap_fwd=timeit(ap_fwd, p, x), ap_vg=timeit(ap_vg, p, x, gout),
                peak=peak_mb())


def main():
    print(f"device {jax.local_devices()[0].device_kind}   "
          f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '(unset)')}")
    print(f"batch {BATCH}, {HEADS} head x {TPH} tables x {ND} det x {NB} buckets, "
          f"{REPS} reps after {WARM} warmup")

    p = M.init(jax.random.PRNGKey(0), NB, ND, TPH, HEADS, OBS, 12, delay_init_std=0.0)
    x = jax.random.normal(jax.random.PRNGKey(3), (BATCH, OBS))
    gout = jax.random.normal(jax.random.PRNGKey(7), (BATCH, HEADS, 12))

    r = {f: measure(f, p, x, gout) for f in ("soft", "detach_hard")}
    M.SPIKE_FORM = "detach_hard"

    print(f"\n  {'scope':<26} {'soft ms':>9} {'detach ms':>10} {'speedup':>9} "
          f"{'soft it/s':>10} {'detach it/s':>12}")
    for key, label in (("fs_fwd", "first_spike  forward"),
                       ("fs_vg", "first_spike  value+grad"),
                       ("ap_fwd", "apply        forward"),
                       ("ap_vg", "apply        value+grad")):
        s, d = r["soft"][key], r["detach_hard"][key]
        print(f"  {label:<26} {s:9.3f} {d:10.3f} {s/d:8.2f}x {1e3/s:10.1f} "
              f"{1e3/d:12.1f}")
    print(f"  {'peak device memory (MB)':<26} {r['soft']['peak']:9.1f} "
          f"{r['detach_hard']['peak']:10.1f}")

    fwd_gain = r["soft"]["fs_fwd"] / r["detach_hard"]["fs_fwd"]
    vg_gain = r["soft"]["fs_vg"] / r["detach_hard"]["fs_vg"]
    print(f"\n  ATTRIBUTION, on the isolated block that differs:")
    print(f"    forward     {fwd_gain:.2f}x  — dropping the sigmoid, cumprod and "
          f"weighted sum over N")
    print(f"    value+grad  {vg_gain:.2f}x  — the same, plus the VJP of the "
          f"cumprod-survival chain")
    if vg_gain > fwd_gain * 1.15:
        print(f"    => the saving is predominantly in the BACKWARD: differentiating "
              f"through the\n       soft crossing costs more than computing it.")
    elif fwd_gain > vg_gain * 1.15:
        print(f"    => the saving is predominantly in the FORWARD.")
    else:
        print(f"    => forward and backward contribute comparably.")

    ap = r["soft"]["ap_vg"] - r["detach_hard"]["ap_vg"]
    print(f"\n  Whole-module value+grad: {r['soft']['ap_vg']:.3f} -> "
          f"{r['detach_hard']['ap_vg']:.3f} ms ({ap:+.3f} ms, "
          f"{r['soft']['ap_vg']/r['detach_hard']['ap_vg']:.2f}x). A SAC iteration runs 32 "
          f"updates,\n  so that is ~{32*ap:.1f} ms/iter against exp_c50's measured "
          f"~220 ms/iter — about {100*32*ap/220:.1f}%.")


if __name__ == "__main__":
    main()
