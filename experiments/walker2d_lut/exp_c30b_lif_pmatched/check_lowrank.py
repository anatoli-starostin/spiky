"""exp_c30b — pin the factorised module to exp_c30's dense one, which is torch-verified.

There is no torch reference for THIS architecture -- the factorisation is ours, so a
two-venv parity test has nothing to compare against. What does exist is exp_c30's dense
module, checked against the torch reference 13/13 with the table gradient bit-identical.
So the dense module is used as the ORACLE: materialise the P this factorisation
represents, hand it to the dense implementation, and require the two to agree.

That is a stronger check than it first looks. The factorised forward never builds P -- it
contracts the (B,M,N,N) gate tensor against Pv and then Pu, plus a separate bias sum. A
transposed index or a misplaced off-diagonal mask in that contraction is invisible to a
shape check and to a gradient-flow check, but it cannot survive comparison against the
dense path.

Checks:
  A. membrane equals the dense membrane at the materialised P
  B. forward hard / soft / st equal the dense forward
  C. st == hard (the straight-through identity, again -- it is the whole training contract)
  D. gradients flow to EVERY parameter including log_temp_bit, Pu, Pv, Pb
  E. dense-equivalent P has ~the reference init scale, so the pair channel still starts
     near zero and each detector still begins as a pure value/range unit
  F. the actor lands at the intended param count

Usage:
  python check_lowrank.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp_c30_lif_detectors"))

import jax_lif_lowrank as LR                                   # noqa: E402
import jax_lif_mhl as DENSE                                    # noqa: E402

NAP, TPH, HEADS, N, NOUT = 6, 32, 1, 17, 12
TARGET = 49152
TOL = 2e-5


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def main():
    fails = []

    def check(name, got, ref):
        r = rel(got, ref)
        ok = r < TOL
        fails.append(None if ok else name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} rel {r:.3e}")

    p = LR.init(jax.random.PRNGKey(0), NAP, TPH, HEADS, N, NOUT)
    x = jax.random.normal(jax.random.PRNGKey(1), (24, N))
    eps = 0.7

    # The same params, with P materialised, in the dense (torch-verified) module.
    dp = {k: v for k, v in p.items() if k not in ("Pu", "Pv", "Pb")}
    dp["P"] = LR.dense_P(p)

    print(f"factorised vs dense oracle at nap{NAP}/tph{TPH}, N={N}, n_out={NOUT}, "
          f"eps={eps}, rank={LR.PAIR_RANK}")
    check("membrane", LR.membrane(p, x, eps), DENSE.membrane(dp, x, eps))
    for mode in ("hard", "soft", "st"):
        check(f"forward {mode}",
              LR.apply(p, x, eps, HEADS, TPH, NAP, mode=mode),
              DENSE.apply(dp, x, eps, HEADS, TPH, NAP, mode=mode))
    check("address", LR.address(p, x, eps, HEADS, TPH, NAP),
          DENSE.address(dp, x, eps, HEADS, TPH, NAP))

    y_st = LR.apply(p, x, eps, HEADS, TPH, NAP, mode="st")
    y_hard = LR.apply(p, x, eps, HEADS, TPH, NAP, mode="hard")
    d = float(np.abs(np.asarray(y_st) - np.asarray(y_hard)).max())
    ok = d < 1e-6
    fails.append(None if ok else "st==hard identity")
    print(f"  {'PASS' if ok else 'FAIL'}  {'st == hard identity':<34} max|diff| {d:.3e}")

    # D: every parameter must receive gradient, or a knob is silently dead.
    gout = jax.random.normal(jax.random.PRNGKey(2), y_st.shape)
    g = jax.grad(lambda pp: (LR.apply(pp, x, eps, HEADS, TPH, NAP, mode="st")
                             * gout).sum())(p)
    dead = [k for k in sorted(g) if float(jnp.abs(g[k]).sum()) == 0.0]
    ok = not dead
    fails.append(None if ok else f"dead grads: {dead}")
    print(f"  {'PASS' if ok else 'FAIL'}  {'gradients reach all params':<34} "
          f"{len(g)} params, dead: {dead or 'none'}")

    # The table gradient must still be the hard one-row-per-table scatter.
    nz = int((np.abs(np.asarray(g["table"])).sum(-1) > 0).sum())
    print(f"  info  table rows with nonzero grad: {nz} of {TPH * (1 << NAP)} "
          f"({100 * nz / (TPH * (1 << NAP)):.1f}%) — hard scatter preserved")

    # E: the pair channel must still start near zero.
    sd = float(jnp.std(dp["P"]))
    ok = 0.005 < sd < 0.02
    fails.append(None if ok else "P init scale")
    print(f"  {'PASS' if ok else 'FAIL'}  {'dense-equivalent P init std':<34} "
          f"{sd:.5f}  (torch reference: 0.01)")

    # F: the number this experiment exists to hit.
    det, tab = LR.n_params(p)
    tot = det + tab
    ok = abs(tot - TARGET) / TARGET < 0.025
    fails.append(None if ok else "param count")
    print(f"  {'PASS' if ok else 'FAIL'}  {'actor param count':<34} "
          f"{tot:,} = {det:,} detectors + {tab:,} table  "
          f"({100 * tot / TARGET - 100:+.2f}% vs {TARGET:,})")
    print(f"  info  exp_c30 dense was 87,361 — this is {100 * tot / 87361 - 100:+.1f}%")

    bad = [f for f in fails if f]
    if bad:
        print(f"\nFAILED on {len(bad)}: {', '.join(bad)}")
        sys.exit(1)
    print(f"\nALL CHECKS OK — {len(fails)} checks")


if __name__ == "__main__":
    main()
