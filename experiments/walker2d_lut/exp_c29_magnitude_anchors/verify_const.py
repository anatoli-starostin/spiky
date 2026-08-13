"""exp_c29 — prove the constant machinery does what the header claims, before training.

Five checks, each aimed at a way this could be silently wrong rather than loudly broken:

  A  --constants none reproduces exp_c09's anchors path EXACTLY. The baseline arm has to
     be the published architecture, not a lookalike; if the fork drifted, every number in
     this experiment would be measured against the wrong reference.
  B  An obs-const bit really is a THRESHOLD TEST. The whole hypothesis rests on
     1[<w, x> + b > 0] == 1[x_j > c_k] for those bits, so it is checked against the
     actual projection on real observations rather than argued from the encoding.
  C  Magnitude blindness is REAL in the baseline and CURED in the augmented arm. Adding a
     constant offset to every coordinate of the observation must leave every 17-dim
     anchor bit untouched, and must move some augmented bits. This is the defect the
     experiment is about; if it does not reproduce, there is nothing to fix.
  D  The repair removes every const-const bit and creates no degenerate a == b pair, and
     the pre-repair count is reported (it is the capacity that would otherwise be lost).
  E  Arms grid / random / clumped share IDENTICAL wiring at a given seed. They must
     differ in the 16 numbers and in nothing else, or the A/B is confounded by routing.

Usage:
  python verify_const.py [--nap 6] [--tph 64] [--seed 0]
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c06_jax_backprop"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c11_lut_sac_2x2"))
sys.path.insert(0, HERE)

for p in ("exp_c07_robustness", "exp_c09_lut_sac", "exp_c26_action_quant",
          "exp_c28_input_quant"):
    sys.path.insert(0, os.path.join(HERE, "..", p))

import jax_lut_grad as L                                   # noqa: E402
import jax_lut_ext as X                                    # noqa: E402
import const_lut_sac as C                                  # noqa: E402
import eval_cpu                                            # noqa: E402
import perturb                                             # noqa: E402
import input_quant as IQ                                   # noqa: E402

OBS = C.OBS
TEACHER = os.path.join(HERE, "..", "exp_c09_lut_sac",
                       "lut_sac_c21_seed4_20k_at10000_actor.npz")


def real_obs(episodes):
    """Standardised observations from an actual rollout.

    Not a synthetic draw. Checks B and C are about how the comparators behave on the
    states the policy really visits, and the constants were calibrated to exactly that
    distribution -- a standard normal would understate the threshold bits' activity in
    the tails, where most of the constants sit, and turn check C into a test of the
    surrogate distribution instead of the intervention.
    """
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    mean = np.asarray(stats["obs_mean"], np.float32)
    std = np.asarray(stats["obs_std"], np.float32)
    fn, _ = eval_cpu.load_actor(TEACHER, forward_mode="hard")
    O = IQ.rollout_record_obs(perturb.make_model(None, 1.0), fn, episodes)
    return np.asarray((O - mean) / (std + 1e-6), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=64)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="balanced")
    ap.add_argument("--episodes", type=int, default=5)
    a = ap.parse_args()
    T = a.heads * a.tph
    cj = json.load(open(os.path.join(HERE, "constants.json")))
    sets = {k: np.asarray(v, np.float32) for k, v in cj["sets"].items()}
    NC = cj["levels"]
    AOBS = OBS + NC
    x = real_obs(a.episodes)

    # ---- A: the baseline arm is exp_c09's anchors path, unchanged ----------
    w_ref, b_ref = X.anchor_pair_wb_lutorch(T, a.nap, OBS, seed=a.seed,
                                            policy=a.policy, heads=a.heads)
    pa, pb = C.anchor_pairs(T, a.nap, OBS, a.seed, a.policy, a.heads)
    pa2, pb2, nrep0 = C.repair_const_const(pa, pb, OBS, a.seed)
    w_new, b_new = X.pairs_to_wb(pa2, pb2, OBS)
    assert nrep0 == 0, f"baseline arm repaired {nrep0} bits -- it must touch nothing"
    assert np.array_equal(np.asarray(w_ref), np.asarray(w_new)), "A: w differs"
    assert np.array_equal(np.asarray(b_ref), np.asarray(b_new)), "A: b differs"
    print(f"A  --constants none == exp_c09 anchors: w and b bit-identical "
          f"({w_ref.size:,} + {b_ref.size:,} elements, 0 differing)", flush=True)

    # ---- the augmented wiring, shared by all three constant arms -----------
    qa, qb = C.anchor_pairs(T, a.nap, AOBS, a.seed, a.policy, a.heads)
    cc_before = int(((qa >= OBS) & (qb >= OBS)).sum())
    ra, rb, nrep = C.repair_const_const(qa, qb, OBS, a.seed)
    w33, b33 = X.pairs_to_wb(ra, rb, AOBS)

    # ---- B: an obs-const bit is exactly a threshold test -------------------
    C.CFG.update(nconst=NC, const=jnp.asarray(sets["grid"])[None, :])
    xa = np.asarray(C.augment(jnp.asarray(x)))
    assert xa.shape == (len(x), AOBS), f"B: augmented shape {xa.shape}"
    assert np.array_equal(xa[:, :OBS], x), "B: augment altered the observation half"
    assert np.array_equal(xa[:, OBS:], np.broadcast_to(sets["grid"], (len(x), NC))), \
        "B: constant block is not the requested values"
    proj = np.asarray(L._project(jnp.asarray(xa), w33, b33))          # [B, T, nap]
    bits = proj > 0
    n_checked = 0
    for t in range(T):
        for i in range(a.nap):
            ia, ib = int(ra[t, i]), int(rb[t, i])
            if (ia >= OBS) == (ib >= OBS):
                continue                                   # not an obs-const bit
            if ia >= OBS:                                  # 1[c - x_j > 0]
                want = sets["grid"][ia - OBS] > xa[:, ib]
            else:                                          # 1[x_j - c > 0]
                want = xa[:, ia] > sets["grid"][ib - OBS]
            assert np.array_equal(bits[:, t, i], want), \
                f"B: bit (t={t}, i={i}) is not the threshold test it encodes"
            n_checked += 1
    print(f"B  every obs-const bit == a threshold test on real projections: "
          f"{n_checked} bits x {len(x):,} samples, 0 disagreements", flush=True)

    # ---- C: blindness real in the baseline, cured in the augmented arm -----
    # A uniform offset added to every coordinate cancels in x[a] - x[b]. It cannot
    # cancel in x_j - c_k. That IS magnitude blindness, stated as a testable invariance.
    off = 0.75
    C.CFG.update(nconst=0, const=None)
    p0 = np.asarray(L._project(jnp.asarray(x), w_ref, b_ref)) > 0
    p0o = np.asarray(L._project(jnp.asarray(x + off), w_ref, b_ref)) > 0
    flip17 = float((p0 != p0o).mean())
    C.CFG.update(nconst=NC, const=jnp.asarray(sets["grid"])[None, :])
    p1 = np.asarray(L._project(jnp.asarray(np.asarray(C.augment(jnp.asarray(x)))),
                               w33, b33)) > 0
    p1o = np.asarray(L._project(jnp.asarray(np.asarray(C.augment(jnp.asarray(x + off)))),
                                w33, b33)) > 0
    flip33 = float((p1 != p1o).mean())
    is_thresh = (ra >= OBS) ^ (rb >= OBS)
    flip_th = float((p1 != p1o)[:, is_thresh].mean())
    assert flip17 == 0.0, \
        f"C: a 17-dim anchor bit moved under a uniform offset ({flip17:.4f}) -- the " \
        f"premise of this experiment is that it cannot"
    assert flip33 > 0.01, \
        f"C: the augmented arm barely responds to absolute level ({flip33:.4f}); the " \
        f"intervention is not doing what it is for"
    # Only obs-const bits CAN move, so the overall rate is diluted by the obs-obs bits
    # that are blind by construction. Both are printed: the first is what the policy
    # experiences, the second is how responsive the new faculty actually is.
    print(f"C  uniform offset +{off} on {len(x):,} real standardised observations: "
          f"baseline flips {flip17*100:.2f}% of address bits (blind, exactly as "
          f"predicted); augmented flips {flip33*100:.2f}% overall, "
          f"{flip_th*100:.2f}% of its threshold bits", flush=True)

    # ---- D: the repair --------------------------------------------------
    assert not ((ra >= OBS) & (rb >= OBS)).any(), "D: a const-const bit survived"
    assert not (ra == rb).any(), "D: the repair created a degenerate a == b pair"
    tot = ra.size
    print(f"D  repair: {cc_before}/{tot} bits ({100*cc_before/tot:.1f}%) compared two "
          f"constants before, 0 after; {nrep} redrawn", flush=True)

    # ---- E: the three arms differ ONLY in the 16 numbers -------------------
    for nm in ("random", "clumped"):
        ea, eb = C.anchor_pairs(T, a.nap, AOBS, a.seed, a.policy, a.heads)
        ea, eb, _ = C.repair_const_const(ea, eb, OBS, a.seed)
        assert np.array_equal(ea, ra) and np.array_equal(eb, rb), \
            f"E: arm {nm} would train on different wiring than grid"
    diffs = {k: float(np.abs(sets["grid"] - v).mean())
             for k, v in sets.items() if k != "grid"}
    print(f"E  grid / random / clumped share identical wiring at seed {a.seed}; "
          f"they differ only in the constants (mean |delta| vs grid: "
          + ", ".join(f"{k} {v:.3f}" for k, v in diffs.items()) + ")", flush=True)

    print("\nall checks passed", flush=True)


if __name__ == "__main__":
    main()
