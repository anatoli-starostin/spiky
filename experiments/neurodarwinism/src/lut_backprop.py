"""exp011 substrate: a FastMultiHeadLut candidate, trained by plain backprop.

The chapter's evolutionary loop has only ever been run on SPNet, where the inner learning step
is STDP and every negative result is confounded by "did the substrate even train?". This module
swaps the substrate for one we trust: the REAL `spiky.lutorch.FastMultiHeadLut` -- the
anchor-pair LUT that IS the distillation teacher -- with the inner step replaced by Adam on
random minibatches of the same dataset.

The genome is the LUT's CAPACITY HYPERPARAMETERS, not its weights. Weights are trained from
scratch for every candidate, so selection sees "how well does this ARCHITECTURE fit, once
trained", which is exactly the question "what is the minimal anchor-pair LUT that fits this
dataset".

    param_count = n_heads * tables_per_head * 2^n_anchor_pairs * n_outputs   (+2 learnable temps)

NOTE ON n_heads. The head axis and the tables_per_head axis are the SAME capacity axis here:
the module sums over tables within a head, and a 6-dim target forces us to reduce the head axis
too, so `n_heads=h, tables_per_head=t` is the same function class as `n_heads=1,
tables_per_head=h*t` with the same parameter count. The genome therefore pins n_heads at 1 by
default and moves capacity on tables_per_head; --evolve-heads re-opens it if you want the
redundancy in the search space.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import load                                       # noqa: E402

N_IN, N_OUT = 17, 6

# HARD FORWARD ONLY, FOR TRAINING AND EVALUATION ALIKE.
#
# The hard forward selects one row per table by sign-packing the pairwise differences, which is
# piecewise constant and has a zero-a.e. true derivative. It is still trainable because
# FastMultiHeadLut ships a SOFT SURROGATE backward: the input and temperature gradients come
# from a full K-row softmax surrogate pinned to the chosen row, while the weight gradient is a
# 1-row scatter reflecting the actual forward. So backprop rides the hard forward pass and the
# surrogate gradient -- there is no need for, and no use of, a smooth forward path here.
#
# Keeping it hard everywhere also means TRAINING AND EVALUATION ARE THE SAME FUNCTION. A
# hybrid_smooth-trained, hard-evaluated candidate would be scored on a network it never
# optimised, and the whole point of exp011 is a substrate whose training we do not have to
# reason about. `lut_grad_check.py` verifies all of this empirically rather than assuming it.
FORWARD_MODE = "hard"

# The capacity/optimisation knobs the genome carries. Values here are the reference mid-size
# config: the teacher's own shape (NAP 6 -> 64 rows, 32 tables, 1 head).
DEFAULT_GENOME = dict(
    n_anchor_pairs=6,          # NAP in [1, 15]; each table has 2^NAP rows
    tables_per_head=32,        # tables summed per head
    n_heads=1,                 # see the module docstring: redundant with tables_per_head
    forward_mode="hard",       # THE ONLY MODE USED -- see FORWARD_MODE below
    learnable_temps=False,
    soft_score_temp=0.5,
    select_temp=0.5,
    lr=3e-3,
    anchor_seed=1,             # which anchor-pair draw this candidate got
)

# NAP is capped at 15 by the module, but 2^NAP * tables_per_head * 6 floats has to fit and be
# trainable in a few thousand steps. 12 gives 4096 rows/table, already 1.5M params at 64 tables.
NAP_RANGE = (1, 12)
TPH_RANGE = (1, 128)
HEADS_RANGE = (1, 8)
LR_RANGE = (3e-4, 3e-2)


def param_count(g):
    n = g["n_heads"] * g["tables_per_head"] * (1 << g["n_anchor_pairs"]) * N_OUT
    return int(n + (2 if g["learnable_temps"] else 0))


def genome_str(g):
    return (f"NAP {g['n_anchor_pairs']:2d} (2^{g['n_anchor_pairs']}={1 << g['n_anchor_pairs']:5d} "
            f"rows) x tph {g['tables_per_head']:3d} x heads {g['n_heads']} "
            f"{g['forward_mode']:14s} lr {g['lr']:.4g}  -> {param_count(g):,} params")


def build(g, device="cuda"):
    """The REAL FastMultiHeadLut. Nothing here is a reimplementation."""
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
    assert g.get("forward_mode", FORWARD_MODE) == FORWARD_MODE, (
        f"exp011 is hard-forward only (see FORWARD_MODE); got {g.get('forward_mode')!r}")
    return FastMultiHeadLut(
        input_dim=N_IN, n_heads=g["n_heads"], n_outputs=N_OUT,
        n_anchor_pairs=g["n_anchor_pairs"], tables_per_head=g["tables_per_head"],
        forward_mode=FORWARD_MODE,
        weight_dtype=torch.float32,
        use_bf16=bool(g.get("use_bf16", True)),
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        soft_score_temp=g["soft_score_temp"], select_temp=g["select_temp"],
        learnable_temps=g["learnable_temps"],
        random_seed=int(g["anchor_seed"]),
        device=torch.device(device))


def _fwd(model, x):
    """[B, n_heads, n_outputs] -> [B, n_outputs]. Sum over heads, matching the within-head
    reduction, so heads and tables compose the same way (see the module docstring)."""
    return model(x).sum(dim=1).float()


@torch.no_grad()
def evaluate(model, X, Y, batch=4096):
    se, n = 0.0, 0
    for i in range(0, X.shape[0], batch):
        p = _fwd(model, X[i:i + batch])
        se += float(((p - Y[i:i + batch]) ** 2).sum())
        n += p.numel()
    return se / n


def train_eval(g, Xtr, Ytr, Xte, Yte, steps=2000, batch=512, seed=0, device="cuda",
               eval_every=0, log=None):
    """Train ONE candidate from scratch and return its held-out MSE and size.

    Weights are always freshly initialised: the genome is the architecture, so a candidate's
    score has to be "what this architecture reaches when trained", not "what these particular
    weights happen to be worth". That is also what makes the score comparable across the pool
    when the pool contains different shapes.
    """
    torch.manual_seed(seed)
    model = build(g, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(g["lr"]))
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    t0 = time.time()
    curve = []
    for s in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, batch)).to(device)
        loss = torch.nn.functional.mse_loss(_fwd(model, Xtr[idx]), Ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if eval_every and ((s + 1) % eval_every == 0 or s == 0):
            m = evaluate(model, Xte, Yte)
            lv = float(loss.detach())
            curve.append(dict(step=s + 1, train_batch_mse=lv, heldout_mse=m))
            if log:
                print(f"      step {s + 1:5d}  batch {lv:.5f}  held-out {m:.5f}", flush=True)
    out = dict(genome={k: (float(v) if isinstance(v, float) else v) for k, v in g.items()},
               params=param_count(g), steps=steps, batch=batch,
               heldout_mse=evaluate(model, Xte, Yte),
               train_mse=evaluate(model, Xtr[:20000], Ytr[:20000]),
               seconds=round(time.time() - t0, 1), curve=curve)
    del model, opt
    torch.cuda.empty_cache()
    return out


def to_device(seed, n_val, device="cuda"):
    """The chapter's split, moved to the GPU once. x_norm -> y_action_mean."""
    _, _, Xp, Yp, Xv, Yv = load(64, seed, n_val)
    t = lambda a: torch.tensor(np.asarray(a, np.float32), device=device)   # noqa: E731
    return t(Xp), t(Yp), t(Xv), t(Yv)


def baselines(Ytr, Yte):
    """What "fits" has to beat: predict the training mean for everything."""
    mu = Ytr.mean(0, keepdim=True)
    return dict(constant_predictor_mse=float(((Yte - mu) ** 2).mean()),
                target_var=float(Yte.var()),
                target_sd=float(Yte.std()))


def main():
    ap = argparse.ArgumentParser(description="exp011 sanity: train single LUT configs")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--nap", type=int, nargs="+", default=None)
    ap.add_argument("--tph", type=int, nargs="+", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Xte, Yte = to_device(a.seed, a.n_val, dev)
    base = baselines(Ytr, Yte)
    print(f"exp011 substrate check: {Xtr.shape[0]:,} train / {Xte.shape[0]:,} held-out states, "
          f"target sd {base['target_sd']:.4f}, dev {dev}")
    print(f"  BASELINE constant predictor held-out MSE {base['constant_predictor_mse']:.5f}\n")

    if a.nap and a.tph:
        grid = list(zip(a.nap, a.tph))
    else:
        # small -> large, spanning three orders of magnitude in parameter count
        grid = [(4, 4), (5, 8), (6, 32), (8, 32), (10, 32)]
    rows = []
    for nap, tph in grid:
        g = dict(DEFAULT_GENOME, n_anchor_pairs=nap, tables_per_head=tph,
                 use_bf16=not a.no_bf16)
        if a.lr:
            g["lr"] = a.lr
        print(f"  {genome_str(g)}")
        r = train_eval(g, Xtr, Ytr, Xte, Yte, a.steps, a.batch, a.seed, dev,
                       eval_every=a.eval_every, log=True)
        r["baselines"] = base
        rows.append(r)
        print(f"    -> held-out MSE {r['heldout_mse']:.5f}  train {r['train_mse']:.5f}  "
              f"({r['params']:,} params, {r['seconds']}s)\n", flush=True)

    print(f"{'NAP':>4} {'tph':>4} {'params':>10} {'held-out MSE':>13} {'vs constant':>12}")
    for r in rows:
        g = r["genome"]
        print(f"{g['n_anchor_pairs']:4d} {g['tables_per_head']:4d} {r['params']:10,} "
              f"{r['heldout_mse']:13.5f} "
              f"{r['heldout_mse'] / base['constant_predictor_mse']:11.3f}x")
    if a.out:
        json.dump(dict(baselines=base, rows=rows), open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
