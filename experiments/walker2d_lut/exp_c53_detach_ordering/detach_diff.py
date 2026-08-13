"""exp_c53 — what actually changes when the spike crossing is detached.

Run in the MJX venv. CPU is fine; this is a semantics check, not a benchmark.

THREE QUESTIONS, and the first one has a subtlety worth stating before the numbers.

1. DOES THE ORDERING GRADIENT EXIST AT ALL? The task frames this variant as "stop_gradient
   on the ordering". But the reorder decision was ALREADY non-differentiable in every
   SORT_FORM: `rank` builds its permutation from integer comparisons, `argsort` from
   integer indices, and exp_c52's sort_equivalence.py measured their gradients identical to
   0.000e+00 on all 8 parameters -- including on an input where 100% of arrival pairs are
   exactly tied. This script re-checks it directly: wrap the permutation in an explicit
   `stop_gradient` and see whether ANY gradient changes. If nothing moves, "detach the
   ordering" cannot be the operative change, and what follows is.

2. HOW OFTEN DO THE ADDRESSES DIFFER? Note what cannot differ: `b_hard` is computed from
   `t_hard` in BOTH variants, and `t_hard` does not depend on T_cross. So the HARD digits
   -- the cell each table actually reads -- are identical by construction, and a "digit
   disagreement rate" between the two variants is necessarily 0. What genuinely differs is
   the SOFT partition `g`, which is fed `t_soft` under the soft variant and `t_hard` under
   this one, and `g` is the entire address-gradient path. So the honest measurement is the
   distance between the two soft partitions, plus how often each variant's soft argmax
   disagrees with the hard digit it is supposed to be a surrogate for.

3. WHAT STOPS LEARNING? `w_raw` and `tau_raw` reach the output only through the membrane
   potential V, and under "detach_hard" V is used only to pick a detached index. The
   parity gate already asserts both are dead on both sides; this counts the parameters.

Usage:
  python detach_diff.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax_mhl_lut as M          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C50 = os.path.join(HERE, "..", "exp_c50_no_delay_clamp")
HEADS, TPH, ND, NB = 1, 128, 1, 16
BATCH, OBS = 256, 17
PKEYS = ("delay", "w_raw", "tau_raw", "beta_base", "beta_raw",
         "log_T_cross", "log_T_bkt", "table")


def under(form, fn):
    keep, M.SPIKE_FORM = M.SPIKE_FORM, form
    try:
        return jax.tree.map(np.asarray, fn())
    finally:
        M.SPIKE_FORM = keep


def grads(p, x, gout):
    return jax.grad(lambda pp: (M.apply(pp, x, HEADS, TPH, NB, ND,
                                        mode="train") * gout).sum())(p)


def q1_ordering_gradient(p, x, gout):
    """Is there any gradient flowing through the permutation, in either variant?"""
    print("\n=== 1. Does the ORDERING carry gradient? "
          "(explicit stop_gradient vs not) ===")
    orig = M._sorted_arrivals

    def detached(a, w):
        """Same permutation, but every path through it explicitly severed."""
        a_srt, w_srt = orig(a, w)
        # Recover the permutation as a hard gather and rebuild the outputs so that the
        # ONLY route from `a` to the result is the gathered value, never the decision.
        idx = jax.lax.stop_gradient(jnp.argsort(a, axis=-1, stable=True))
        wb = jnp.broadcast_to(w[None], a.shape)
        return (jnp.take_along_axis(a, idx, axis=-1),
                jnp.take_along_axis(wb, idx, axis=-1))

    for form in ("soft", "detach_hard"):
        base = under(form, lambda: grads(p, x, gout))
        M._sorted_arrivals = detached
        try:
            det = under(form, lambda: grads(p, x, gout))
        finally:
            M._sorted_arrivals = orig
        worst = max(float(np.abs(base[k].astype(np.float64)
                                 - det[k].astype(np.float64)).max()) for k in base)
        print(f"  SPIKE_FORM={form:<12} max |grad(normal) - grad(stop_gradient on the "
              f"permutation)| = {worst:.3e}"
              + ("   <- the ordering was ALREADY detached" if worst == 0.0
                 else "   <- the ordering DID carry gradient"))


def q2_addresses(p, x):
    print("\n=== 2. How far apart are the two variants' addresses? ===")

    def parts():
        t_hard, t_soft = M.first_spike(p, x)
        b_hard, g = M.bucket(p, t_hard, t_soft)
        return dict(t_hard=t_hard, t_soft=t_soft, b_hard=b_hard, g=g,
                    y=M.apply(p, x, HEADS, TPH, NB, ND, mode="train"))

    s = under("soft", parts)
    d = under("detach_hard", parts)

    nd = int((s["b_hard"] != d["b_hard"]).sum())
    print(f"  HARD digits differ                 {nd:,} of {s['b_hard'].size:,} "
          f"({100*nd/s['b_hard'].size:.4f}%)   "
          f"<- 0 by construction: b_hard reads t_hard in both")
    print(f"  t_soft vs t_hard, mean |gap|       "
          f"{float(np.abs(s['t_soft'] - s['t_hard']).mean()):.4f}  "
          f"(max {float(np.abs(s['t_soft'] - s['t_hard']).max()):.4f}) "
          f"<- how far the soft crossing sat from the real one")
    print(f"  soft partition g, max |difference| "
          f"{float(np.abs(s['g'] - d['g']).max()):.4f}")
    print(f"  soft partition g, mean |difference|"
          f" {float(np.abs(s['g'] - d['g']).mean()):.6f}")
    for nm, r in (("soft", s), ("detach_hard", d)):
        am = np.asarray(r["g"]).argmax(-1)
        agree = float((am == np.asarray(r["b_hard"])).mean())
        print(f"  [{nm:<11}] argmax(g) == b_hard   {100*agree:.2f}%   "
              f"<- how faithful the soft surrogate is to the hard address it stands in for")
    print(f"  train forward, max |difference|    "
          f"{float(np.abs(s['y'] - d['y']).max()):.6f}   "
          f"<- ST value is the hard read, so this is the address-path term only")


def q3_dead(p, x, gout):
    print("\n=== 3. What stops learning? ===")
    for form in ("soft", "detach_hard"):
        g = under(form, lambda: grads(p, x, gout))
        dead = {k: int(np.asarray(g[k]).size) for k in sorted(g)
                if float(np.abs(np.asarray(g[k])).max()) == 0.0}
        tot = sum(int(np.asarray(v).size) for v in g.values())
        nd = sum(dead.values())
        print(f"  SPIKE_FORM={form:<12} {nd:,} of {tot:,} parameters receive exactly "
              f"zero gradient ({100*nd/tot:.1f}%)"
              + (f"\n{'':<28}dead: " + ", ".join(f"{k} ({v:,})"
                                                 for k, v in dead.items()) if dead else ""))


def main():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(jax.random.PRNGKey(3), (BATCH, OBS))
    gout = jax.random.normal(jax.random.PRNGKey(7), (BATCH, HEADS, 12))

    f = os.path.join(C50, "mhl_sac_c50_s0_actor.npz")
    if os.path.exists(f):
        z = np.load(f)
        p = {k: jnp.asarray(z[k]) for k in PKEYS if k in z.files}
        label = "TRAINED exp_c50 seed 0"
    else:
        p = M.init(key, NB, ND, TPH, HEADS, OBS, 12, delay_init_std=0.0)
        label = "fresh init"
    print(f"weights: {label}   batch {BATCH}   "
          f"{HEADS} head x {TPH} tables x {ND} det x {NB} buckets")

    q1_ordering_gradient(p, x, gout)
    q2_addresses(p, x)
    q3_dead(p, x, gout)


if __name__ == "__main__":
    main()
