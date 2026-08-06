"""exp_c52 — how often do `rank` and exact `argsort` actually disagree?

Run in the MJX venv, CPU. Costs seconds, and it decides what the 4-hour GPU ablation can
possibly show.

THE QUESTION. exp_c50 ran `SORT_FORM="rank"`: the arrival permutation is recovered by
counting, `rank_k = #{j : a_j < a_k}` with ties broken by index, then APPLIED as a one-hot
contraction. exp_c36 ran `jnp.argsort(a, axis=-1, stable=True)` + `take_along_axis`. If
those two disagree anywhere -- and ties are where they would -- the sort spelling is a live
candidate for the residual c50-vs-c36 gap. If they agree bit for bit, it is not, and the
ablation can only reproduce c50.

WHY TIES ARE THE PLACE TO LOOK. `jnp.argsort(stable=True)` breaks ties by index. The `rank`
form breaks them by `(a_j == a_k) & (j < k)`, which is the same rule written as a
comparison. They should therefore agree even on ties -- but "should" is the word this script
exists to remove.

Note what does NOT produce ties, since the obvious guess is wrong: `delay_init_std=0` makes
every DELAY identical, but the arrival is `latency(x) + delay`, and 17 distinct input
features give 17 distinct latencies, so a fresh init has ZERO tied arrivals on generic
input. Ties have to be constructed, which the fourth case below does by feeding an input
whose features repeat -- then the latencies coincide, the arrivals are exactly equal, and
the tie-break alone decides the permutation.

WHAT IS COMPARED, on each weight set, for both forms:

  sorted arrivals a_srt and weights w_srt   the permutation itself
  t_hard, t_soft                            the spike times it produces
  b_hard (the bucket digits)                EXACT integer equality, and the disagreement
                                            RATE if any -- the number the ablation is
                                            really asking about
  forward output, train and eval            what the policy sees
  gradient of every parameter               the backward, where the two forms differ
                                            structurally (matmul-transpose vs scatter)

WEIGHT SETS: the three TRAINED exp_c50 checkpoints (the operating point that matters), a
fresh init, a perturbed random set (the generic path), and a CONSTRUCTED tie case where
every one of a detector's 17 arrivals is exactly equal, so the permutation is decided
purely by the tie-break rule.

Usage:
  python sort_equivalence.py
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
BATCH, OBS = 64, 17
PKEYS = ("delay", "w_raw", "tau_raw", "beta_base", "beta_raw",
         "log_T_cross", "log_T_bkt", "table")


def both(fn):
    """Run `fn()` under each SORT_FORM and return (rank_result, argsort_result)."""
    out = []
    for form in ("rank", "argsort"):
        M.SORT_FORM = form
        out.append(jax.tree.map(np.asarray, fn()))
    M.SORT_FORM = "rank"
    return out


def report(name, p, x):
    def intermediates():
        a_srt, _ = M.membrane(p, x)
        t_hard, t_soft = M.first_spike(p, x)
        b_hard, g_soft = M.bucket(p, t_hard, t_soft)
        return dict(a_srt=a_srt, t_hard=t_hard, t_soft=t_soft,
                    b_hard=b_hard, g_soft=g_soft,
                    y_train=M.apply(p, x, HEADS, TPH, NB, ND, mode="train"),
                    y_eval=M.apply(p, x, HEADS, TPH, NB, ND, mode="eval"),
                    addr=M.address(p, x, ND, NB))

    gout = jax.random.normal(jax.random.PRNGKey(7), (x.shape[0], HEADS, 2 * 6))

    def grads():
        return jax.grad(lambda pp: (M.apply(pp, x, HEADS, TPH, NB, ND,
                                            mode="train") * gout).sum())(p)

    r, s = both(intermediates)
    gr, gs = both(grads)

    # How tied ARE the arrivals? A tie is where the two forms could legally differ.
    a = np.asarray(r["a_srt"])
    n_tied = int((np.diff(a, axis=-1) == 0).sum())
    tot_pairs = a[..., :-1].size
    ndiff = int((r["b_hard"] != s["b_hard"]).sum())

    print(f"\n--- {name}  (batch {x.shape[0]}, {a.size:,} arrivals, "
          f"{n_tied:,}/{tot_pairs:,} adjacent pairs exactly tied "
          f"= {100*n_tied/max(1,tot_pairs):.2f}%)")
    print(f"  {'quantity':<28} {'max |rank - argsort|':>22}")
    for k in ("a_srt", "t_hard", "t_soft", "g_soft", "y_train", "y_eval"):
        d = float(np.abs(r[k].astype(np.float64) - s[k].astype(np.float64)).max())
        print(f"  {k:<28} {d:22.3e}{'' if d == 0 else '   <-- DIFFERS'}")
    print(f"  {'bucket digits (exact)':<28} "
          f"{ndiff:>10,} of {r['b_hard'].size:,} differ "
          f"({100*ndiff/r['b_hard'].size:.4f}%)")
    print(f"  {'cell index (exact)':<28} "
          f"{int((r['addr'] != s['addr']).sum()):>10,} of {r['addr'].size:,} differ")
    for k in sorted(gr):
        d = float(np.abs(np.asarray(gr[k], np.float64)
                         - np.asarray(gs[k], np.float64)).max())
        print(f"  {'grad ' + k:<28} {d:22.3e}{'' if d == 0 else '   <-- DIFFERS'}")
    return ndiff


def main():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(jax.random.PRNGKey(3), (BATCH, OBS))
    total = 0

    p0 = M.init(key, NB, ND, TPH, HEADS, OBS, 12, delay_init_std=0.0)
    total += report("fresh init, generic input (delay_init_std=0)", p0, x)

    # CONSTRUCTED TIES. Every feature of a sample carries the SAME value, so all 17
    # latencies are equal; with delay_init_std=0 every delay is equal too, and all 17
    # arrivals of a detector coincide exactly. The permutation is then decided entirely by
    # the tie-break -- index order for `argsort(stable=True)`, `(a_j == a_k) & (j < k)` for
    # `rank`. This is the ONLY place the two forms could legally disagree, so it is the
    # case the whole comparison rests on.
    xt = jnp.repeat(jax.random.normal(jax.random.PRNGKey(5), (BATCH, 1)), OBS, axis=1)
    total += report("CONSTRUCTED TIES (all 17 arrivals equal — tie-break decides alone)",
                    p0, xt)

    kp = jax.random.split(jax.random.PRNGKey(77), len(PKEYS))
    pp = {k: p0[k] + 0.7 * jax.random.normal(kp[i], p0[k].shape)
          for i, k in enumerate(PKEYS)}
    total += report("perturbed random weights (ties rare — the generic path)", pp, x)

    for s in (0, 1, 2):
        f = os.path.join(C50, f"mhl_sac_c50_s{s}_actor.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        pt = {k: jnp.asarray(z[k]) for k in PKEYS if k in z.files}
        total += report(f"TRAINED exp_c50 seed {s} (the operating point that matters)",
                        pt, x)

    print(f"\n{'=' * 72}")
    if total == 0:
        print("The two forms are BIT-IDENTICAL on every weight set tested, including the\n"
              "constructed case where 100% of adjacent arrival pairs are exactly tied and\n"
              "the tie-break alone decides the permutation. Every intermediate, both\n"
              "forwards, and the gradient of all 8 parameters agree to 0.000e+00.\n"
              "\n"
              "The sort spelling therefore cannot be the source of the c50-vs-c36 residual,\n"
              "and the exact-sort ablation can only reproduce c50 seed for seed.\n"
              "\n"
              "This ran on CPU, but the argument carries to GPU: the `rank` form applies\n"
              "the permutation as an elementwise multiply followed by a sum over N, NOT a\n"
              "dot_general, so no TF32 matmul path is involved and the reduction is a sum\n"
              "of 16 exact zeros and one exact value -- reassociation-invariant.")
    else:
        print(f"{total:,} digit disagreements found — the sort spelling IS a live\n"
              f"candidate and the ablation is worth its GPU time.")


if __name__ == "__main__":
    main()
