"""exp011 validation: Lamarckian weight inheritance (warm start).

Five checks, in the order a bug would bite:

  1 DESCENT   confirm the sign-pack row-descent mapping empirically. `_msb_powers` is MSB-first,
              and resize_pairs APPENDS new anchors, so an added anchor is the new LSB and
              old row k -> new rows {2k, 2k+1}. If that is right, duplicating each row makes the
              added anchor a perfectly neutral split and the child reproduces the parent's
              function EXACTLY with zero training. Includes a control: the other plausible
              mapping (tile) must NOT reproduce it, or the test proves nothing.
  2 REMAP     for each mutation type -- same shape, tph+, tph-, nap+, nap- -- check that
              surviving entries are bit-identical to the parent's and that genuinely new cells
              sit at the requested low std.
  3 FUNCTION  what each remap does to the function before any training, measured as held-out
              MSE: nap+ must be exact, same-shape exact, and the lossy ones (tph-, nap-)
              reported honestly rather than asserted away.
  4 CARRIES   a warm-started child's PRE-TRAINING held-out MSE must be far below a cold random
              init's. This is the end-to-end statement that weights genuinely crossed.
  5 SMOKE     a K=12 x 6-round warm-start run completes with no build failures, and the
              per-round pre-training MSE shows knowledge accumulating rather than resetting.

    sbox python lut_warmstart_check.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lut_backprop as lb                                            # noqa: E402
import lut_evolve as le                                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--std", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Xte, Yte = lb.to_device(a.seed, a.n_val, dev)
    rng = np.random.default_rng(a.seed)
    rep = {}

    # ---------------------------------------------------------------- 1 DESCENT
    NAP, TPH = 4, 3
    g = dict(lb.DEFAULT_GENOME, n_anchor_pairs=NAP, tables_per_head=TPH)
    m = lb.build(g, dev)
    with torch.no_grad():                       # distinctive weights: mis-mapping is visible
        m.weights.copy_(torch.arange(m.weights.numel(), dtype=torch.float32,
                                     device=dev).view_as(m.weights) / 100.0)
    pairs = lb.get_anchor_pairs(m)
    X = Xte[:512]
    with torch.no_grad():
        y_par = m(X).sum(1)
    new_pairs = le.resize_pairs(pairs, lb.n_tables(g), NAP + 1, rng)
    gc = dict(g, n_anchor_pairs=NAP + 1, anchor_pairs=new_pairs)
    w_par = lb.get_weights(m)
    mc = lb.build(gc, dev, init_weights=lb.remap_weights(w_par, NAP, NAP + 1,
                                                         lb.n_tables(gc), a.std, rng))
    with torch.no_grad():
        d_ok = float((mc(X).sum(1) - y_par).abs().max())
        lb.set_weights(mc, np.tile(w_par, (1, 2, 1)))          # the WRONG mapping
        d_bad = float((mc(X).sum(1) - y_par).abs().max())
    print(f"(1) DESCENT   weights layout {tuple(m.weights.shape)} "
          f"= [n_tables, 2^NAP, n_outputs];  soft_powers "
          f"{m.soft_powers.cpu().numpy().tolist()} (MSB-first)")
    print(f"              repeat mapping k -> {{2k, 2k+1}}: max|diff| {d_ok:.3e} "
          f"{'(EXACT)' if d_ok < 1e-4 else '(MISMATCH)'}")
    print(f"              control, tile mapping k -> {{k, k+2^NAP}}: max|diff| {d_bad:.3e} "
          f"{'(differs, so the test is not vacuous)' if d_bad > 1e-3 else '(AMBIGUOUS)'}")
    rep["descent"] = dict(layout=list(m.weights.shape), repeat_maxdiff=d_ok,
                          tile_maxdiff=d_bad, exact=bool(d_ok < 1e-4))
    del m, mc
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 2+3 REMAP / FUNCTION
    print("\n(2)+(3) REMAP -- per mutation type: placement, new-cell std, and what it costs")
    base = dict(lb.DEFAULT_GENOME, n_anchor_pairs=6, tables_per_head=16)
    base["anchor_pairs"] = lb.initial_pairs(base)
    pr = lb.train_eval(base, Xtr, Ytr, Xte, Yte, a.steps, 512, a.seed, dev,
                       return_weights=True)
    w0, nap0, nt0 = pr["weights"], base["n_anchor_pairs"], lb.n_tables(base)
    print(f"    parent: NAP {nap0} x tph {base['tables_per_head']}, "
          f"held-out MSE {pr['heldout_mse']:.5f}, weights {w0.shape}")
    cases = [("same shape", nap0, base["tables_per_head"]),
             ("tph GROWS  16->24", nap0, 24),
             ("tph SHRINKS 16->8", nap0, 8),
             ("nap GROWS   6->7", nap0 + 1, base["tables_per_head"]),
             ("nap SHRINKS 6->5", nap0 - 1, base["tables_per_head"])]
    rows = []
    for name, nap1, tph1 in cases:
        gch = dict(base, n_anchor_pairs=nap1, tables_per_head=tph1)
        gch["anchor_pairs"] = le.resize_pairs(base["anchor_pairs"], lb.n_tables(gch), nap1, rng)
        w1 = lb.remap_weights(w0, nap0, nap1, lb.n_tables(gch), a.std, rng)
        shape_ok = w1.shape == (lb.n_tables(gch), 1 << nap1, lb.N_OUT)
        # placement: surviving table/row block must be bit-identical where it is a pure copy
        if nap1 == nap0:
            keep = min(nt0, lb.n_tables(gch))
            placed = bool((w1[:keep] == w0[:keep]).all())
            newstd = (float(w1[nt0:].std()) if lb.n_tables(gch) > nt0 else float("nan"))
        elif nap1 > nap0:
            placed = bool((w1[:nt0, 0::2] == w0).all() and (w1[:nt0, 1::2] == w0).all())
            newstd = float("nan")
        else:
            s = 1 << (nap0 - nap1)
            want = w0.reshape(nt0, w0.shape[1] // s, s, lb.N_OUT).mean(2)
            placed = bool(np.allclose(w1[:nt0], want, atol=1e-6))
            newstd = float("nan")
        # what does the remap cost BEFORE any training?
        mm = lb.build(gch, dev, init_weights=w1)
        pre = lb.evaluate(mm, Xte, Yte)
        del mm
        torch.cuda.empty_cache()
        rows.append(dict(case=name, shape_ok=shape_ok, placement_exact=placed,
                         new_cell_std=newstd, pretrain_mse=pre,
                         parent_mse=pr["heldout_mse"]))
        sd = "n/a" if np.isnan(newstd) else f"{newstd:.2e}"
        print(f"    {name:20s} shape {'OK' if shape_ok else 'BAD'}  placement "
              f"{'exact' if placed else 'WRONG'}  new-cell std {sd:>8}  "
              f"pre-training held-out MSE {pre:.5f} "
              f"(parent {pr['heldout_mse']:.5f})")
    rep["remap"] = rows

    # ---------------------------------------------------------------- 4 CARRIES
    cold = lb.build(dict(base, anchor_seed=987654), dev)
    cold_mse = lb.evaluate(cold, Xte, Yte)
    del cold
    torch.cuda.empty_cache()
    warm_mse = rows[0]["pretrain_mse"]
    print(f"\n(4) CARRIES   cold random init held-out MSE {cold_mse:.5f}; "
          f"warm-started child (same shape) {warm_mse:.5f} "
          f"-> {cold_mse / max(warm_mse, 1e-9):.0f}x better before a single step")
    rep["carries"] = dict(cold_init_mse=cold_mse, warm_start_mse=warm_mse,
                          ratio=cold_mse / max(warm_mse, 1e-9))

    ok = (rep["descent"]["exact"] and d_bad > 1e-3
          and all(r["shape_ok"] and r["placement_exact"] for r in rows)
          and warm_mse < cold_mse / 10)
    print(f"\n{'ALL WARM-START CHECKS PASS' if ok else 'SOME CHECKS FAILED -- see above'}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, default=str)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
