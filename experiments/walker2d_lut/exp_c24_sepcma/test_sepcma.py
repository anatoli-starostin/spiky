"""exp_c24 — verify sep-CMA-ES on standard test functions before it touches Walker2d.

The failure mode this guards against is the one exp_c05 actually shipped: an optimiser
that runs, produces a curve that climbs, and is not the algorithm it claims to be. A
wrong CMA-ES does not crash -- it just behaves like a worse optimiser, which on a hard RL
task is indistinguishable from "the task is hard".

So it is checked against functions with known optima first:

  sphere      f = sum(x^2)                       trivial; catches gross sign/scale errors
  ellipsoid   f = sum(1e6^((i-1)/(n-1)) x_i^2)   condition number 1e6, AXIS-ALIGNED --
                                                 this is what a diagonal covariance is
                                                 FOR, and an isotropic ES cannot do it
  rosenbrock  f = sum(100(x_{i+1}-x_i^2)^2 + (1-x_i)^2)   curved, non-separable valley;
                                                 the honest hard case for sep-CMA-ES

The ellipsoid test is the one that actually proves the covariance update works: with
C fixed at the identity the same loop stalls, so a pass there cannot be explained by the
step-size control alone. That comparison is run explicitly rather than asserted by eye.

Run (CPU is fine and faster for these):
  JAX_PLATFORMS=cpu python test_sepcma.py
"""
import math
import sys

import jax
import jax.numpy as jnp

import sep_cma


def sphere(x):
    return jnp.sum(x ** 2, axis=-1)


def ellipsoid(x):
    n = x.shape[-1]
    powers = jnp.arange(n) / max(n - 1, 1)
    return jnp.sum((1e6 ** powers) * x ** 2, axis=-1)


def rosenbrock(x):
    a = x[..., :-1]
    b = x[..., 1:]
    return jnp.sum(100.0 * (b - a ** 2) ** 2 + (1.0 - a) ** 2, axis=-1)


def optimise(f, n, gens, seed=0, lam=None, sigma0=0.3, x0=None, freeze_C=False):
    st = sep_cma.init(n, lam=lam, sigma0=sigma0,
                      x0=jnp.full((n,), 1.0) if x0 is None else x0)
    key = jax.random.PRNGKey(seed)
    best = float("inf")
    for _ in range(gens):
        key, k = jax.random.split(key)
        x, _z = sep_cma.ask(st, k)
        fit = f(x)
        best = min(best, float(jnp.min(fit)))
        new = sep_cma.tell(st, x, fit, maximise=False)
        if freeze_C:                       # ablation: step size only, no covariance
            new = dict(new, C=st["C"])
        st = new
    return best, st


def check(name, got, target, extra=""):
    ok = got <= target
    print(f"  {name:<44} best {got:>12.3e}  (target < {target:.0e})  "
          f"{'OK' if ok else 'FAIL'}  {extra}")
    return ok


def main():
    print("sep-CMA-ES verification (Ros & Hansen 2008)")
    allok = True

    print("\n1. SPHERE  n=10, lambda=default")
    b, st = optimise(sphere, 10, 300)
    allok &= check("sphere n=10", b, 1e-12, f"sigma ended {st['sigma']:.2e}")

    print("\n2. SPHERE  n=100 (larger d, still easy)")
    b, _ = optimise(sphere, 100, 1500)
    allok &= check("sphere n=100", b, 1e-10)

    print("\n3. ELLIPSOID  n=20, condition 1e6 -- the diagonal-covariance test")
    b_on, st_on = optimise(ellipsoid, 20, 3000)
    allok &= check("ellipsoid n=20, covariance ON", b_on, 1e-8)
    b_off, _ = optimise(ellipsoid, 20, 3000, freeze_C=True)
    print(f"  {'ellipsoid n=20, covariance FROZEN (ablation)':<44} best {b_off:>12.3e}")
    # b_on can reach exactly 0 on this problem, which makes a ratio meaningless rather
    # than impressive -- report the pair and only quote a ratio when one exists.
    if b_on <= 0.0:
        ratio = float("inf")
        print(f"\n  Learning the diagonal takes this to EXACTLY ZERO; frozen it stalls "
              f"at {b_off:.2e}.")
    else:
        ratio = b_off / b_on
        print(f"\n  Learning the diagonal is worth {ratio:.3g}x here "
              f"({b_off:.2e} -> {b_on:.2e}).")
    if ratio > 1e3:
        print("  -> The covariance update is doing real work; this result cannot be "
              "explained\n     by the step-size control alone.")
    else:
        print("  -> WARNING: freezing the covariance barely hurt. Either the update is "
              "not\n     working or this problem does not need it. Investigate before "
              "trusting it.")
        allok = False
    C = st_on["C"]
    # the ellipsoid's optimal variances are proportional to 1/coefficient, so C should
    # span roughly the inverse of the 1e6 condition number
    span = float(C.max() / C.min())
    print(f"  Learned diagonal spans {span:.3e} (the problem's condition number is 1e6)")

    print("\n4. ROSENBROCK  n=10 -- curved, non-separable (the hard, honest case)")
    b, _ = optimise(rosenbrock, 10, 6000, lam=20, sigma0=0.5,
                    x0=jnp.zeros(10))
    # sep-CMA-ES is NOT expected to match full CMA-ES here: the valley is not axis
    # aligned, which is exactly the geometry a diagonal covariance cannot represent.
    # Ros & Hansen report a slowdown on non-separable problems, so the bar is "clearly
    # optimises", not "solves to machine precision".
    allok &= check("rosenbrock n=10", b, 1e-2, "(loose bar on purpose -- see comment)")

    print("\n5. INVARIANCE: maximise flag must mirror minimise exactly")
    b_min, _ = optimise(sphere, 8, 200, seed=3)
    st = sep_cma.init(8, sigma0=0.3, x0=jnp.ones(8))
    key = jax.random.PRNGKey(3)
    best_max = -float("inf")
    for _ in range(200):
        key, k = jax.random.split(key)
        x, _ = sep_cma.ask(st, k)
        fit = -sphere(x)                      # same problem, sign flipped
        best_max = max(best_max, float(jnp.max(fit)))
        st = sep_cma.tell(st, x, fit, maximise=True)
    err = abs((-best_max) - b_min)
    ok = err < 1e-12 * max(abs(b_min), 1e-12) + 1e-15
    print(f"  {'minimise f == maximise -f':<44} {b_min:.6e} vs {-best_max:.6e}  "
          f"{'OK' if ok else 'FAIL'}")
    allok &= ok

    print("\n" + ("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
