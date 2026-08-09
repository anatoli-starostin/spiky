"""The distillation dataset: where it lives and how a run draws batches from it.

Split out of the retired `es_smoke.py` on the move into `src/`. The two functions below are
the ONLY entry points any experiment uses to touch the teacher data, and both are pure
numpy -- no torch, no GPU.

The teacher is the exp19 `FastMultiHeadLut(exp_outputs=True, exp_outputs_scale="sum")`
Walker2d actor; see `../data/README.md` for exactly what the arrays are and how they were
collected. Runs are trained against `y_action_mean` (the full readout) and index the net
with `x_norm` (the normalised observation the LUT itself sees).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.environ.get("ND_NPZ", os.path.join(HERE, "..", "data", "distill_exp19_100k.npz"))


def load(n, seed=0, n_val=2000):
    """-> fixed reference batch (X,Y), the whole training pool, and a FIXED val set.

    SEEDING SCHEME, so runs are reproducible:
      * the fixed reference batch (used for the null baseline and the smoke gates) is
        drawn with numpy default_rng(seed).
      * the FIXED val set is drawn from the last 4000 samples with default_rng(seed + 1).
      * each generation g draws its own training batch with default_rng(seed*1000 + g)
        -- see sample_batch(). One draw per generation, SHARED across the population, so
        candidates stay perfectly paired within a generation.

    The last 4000 samples are held out and NEVER enter the training pool, so the val score
    is a genuine held-out number rather than a resample of what selection already saw.
    """
    Z = np.load(NPZ)
    N = Z["x_norm"].shape[0]
    n_tr = N - 4000
    Xp = Z["x_norm"][:n_tr].astype(np.float64)
    Yp = Z["y_action_mean"][:n_tr].astype(np.float64)
    ref = np.random.default_rng(seed).choice(n_tr, n, replace=False)
    vi = np.random.default_rng(seed + 1).choice(np.arange(n_tr, N), n_val, replace=False)
    return (Xp[ref], Yp[ref], Xp, Yp,
            Z["x_norm"][vi].astype(np.float64), Z["y_action_mean"][vi].astype(np.float64))


def sample_batch(Xp, Yp, n, seed, gen):
    """Fresh training batch for generation `gen`, deterministic in (seed, gen)."""
    idx = np.random.default_rng(seed * 1000 + gen).choice(Xp.shape[0], n, replace=False)
    return Xp[idx], Yp[idx], idx
