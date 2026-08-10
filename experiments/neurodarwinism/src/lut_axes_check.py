"""exp011: does the expanded (nap, tph, anchoring) genome always produce a buildable, trainable
config?

Adding a categorical anchoring gene means mutation can now jump between two different anchor
DRAWING RULES, each with its own constraint (`n_anchor_pairs <= C(input_dim, 2)`). Before
trusting a long evolutionary run, exercise the space directly rather than hoping the smoke test
happens to visit it:

  A  EXHAUSTIVE CORNERS -- every (policy x NAP-extreme x tph-extreme) combination builds and
     takes a training step without raising.
  B  RANDOM WALK -- seed a genome, apply N rounds of the real mutate() operator, and build
     every genome it produces. This is the operator the run will actually use, so it is the
     one that has to be safe, including when it flips the anchoring gene.
  C  HEAD TO HEAD -- the two policies at matched (NAP, tph), several seeds, to see whether
     either is actually better before reading anything into what selection does.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lut_backprop as lb                                         # noqa: E402
import lut_evolve as le                                          # noqa: E402


def try_build_and_step(g, Xtr, Ytr, dev):
    """Build + one optimiser step. Returns None on success, else the exception string."""
    try:
        m = lb.build(g, dev)
        opt = torch.optim.Adam(m.parameters(), lr=float(g["lr"]))
        loss = torch.nn.functional.mse_loss(m(Xtr[:128]).sum(dim=1).float(), Ytr[:128])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ok = bool(torch.isfinite(loss))
        del m, opt
        torch.cuda.empty_cache()
        return None if ok else "non-finite loss"
    except Exception as e:                                        # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", type=int, default=200, help="mutation steps in check B")
    ap.add_argument("--steps", type=int, default=1500, help="training steps in check C")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per arm in check C")
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Xte, Yte = lb.to_device(a.seed, a.n_val, dev)
    rep = dict(anchor_policies=list(lb.ANCHOR_POLICIES))
    print(f"expanded-genome check. Anchoring options enumerated from the implementation: "
          f"{list(lb.ANCHOR_POLICIES)}\n")

    # ---------------------------------------------------------------- A: corners
    naps = [lb.NAP_RANGE[0], 6, lb.NAP_RANGE[1]]
    tphs = [lb.TPH_RANGE[0], 32, lb.TPH_RANGE[1]]
    corners = list(itertools.product(lb.ANCHOR_POLICIES, naps, tphs))
    fails = []
    for pol, nap, tph in corners:
        g = dict(lb.DEFAULT_GENOME, anchor_policy=pol, n_anchor_pairs=nap,
                 tables_per_head=tph, anchor_seed=7)
        err = try_build_and_step(g, Xtr, Ytr, dev)
        if err:
            fails.append(dict(policy=pol, nap=nap, tph=tph, error=err))
    print(f"(A) CORNERS   {len(corners)} combinations "
          f"({len(lb.ANCHOR_POLICIES)} policies x NAP {naps} x tph {tphs}): "
          f"{len(corners) - len(fails)} built and stepped, {len(fails)} failed")
    for f in fails:
        print(f"      FAIL {f}")
    rep["corners"] = dict(n=len(corners), failures=fails)

    # ---------------------------------------------------------------- B: random walk
    rng = np.random.default_rng(a.seed)
    g = le.seed_genome(np.random.default_rng(a.seed))
    walk_fails, seen_pol, naps_seen, tphs_seen = [], set(), set(), set()
    flips = 0
    for i in range(a.walk):
        prev = g["anchor_policy"]
        g = le.mutate(g, rng)
        flips += int(g["anchor_policy"] != prev)
        seen_pol.add(g["anchor_policy"])
        naps_seen.add(g["n_anchor_pairs"])
        tphs_seen.add(g["tables_per_head"])
        err = try_build_and_step(g, Xtr, Ytr, dev)
        if err:
            walk_fails.append(dict(step=i, genome=dict(g), error=err))
    print(f"\n(B) WALK      {a.walk} real mutate() steps: "
          f"{a.walk - len(walk_fails)} built and stepped, {len(walk_fails)} failed")
    print(f"              visited NAP {min(naps_seen)}..{max(naps_seen)} "
          f"({len(naps_seen)} values), tph {min(tphs_seen)}..{max(tphs_seen)} "
          f"({len(tphs_seen)} values), policies {sorted(seen_pol)}, "
          f"{flips} anchoring flips")
    for f in walk_fails[:5]:
        print(f"      FAIL {f}")
    rep["walk"] = dict(steps=a.walk, failures=walk_fails, anchor_flips=flips,
                       nap_range=[int(min(naps_seen)), int(max(naps_seen))],
                       tph_range=[int(min(tphs_seen)), int(max(tphs_seen))],
                       policies_visited=sorted(seen_pol))

    # ---------------------------------------------------------------- C: head to head
    print(f"\n(C) HEAD TO HEAD  the two policies at matched shape, {a.seeds} seeds, "
          f"{a.steps} steps")
    h2h = {}
    for nap, tph in ((6, 32), (5, 64)):
        for pol in lb.ANCHOR_POLICIES:
            ms = []
            for s in range(a.seeds):
                g = dict(lb.DEFAULT_GENOME, anchor_policy=pol, n_anchor_pairs=nap,
                         tables_per_head=tph, anchor_seed=100 + s)
                ms.append(lb.train_eval(g, Xtr, Ytr, Xte, Yte, a.steps, 512, s,
                                        dev)["heldout_mse"])
            h2h[f"NAP{nap}xtph{tph}/{pol}"] = ms
            print(f"      NAP {nap:2d} x tph {tph:3d}  {pol:24s} "
                  f"held-out MSE {np.mean(ms):.5f} +/- {np.std(ms):.5f}  "
                  f"({', '.join(f'{m:.5f}' for m in ms)})")
    rep["head_to_head"] = h2h

    ok = not fails and not walk_fails
    print(f"\n{'ALL BUILD CHECKS PASS' if ok else 'BUILD FAILURES PRESENT -- see above'}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, default=str)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
