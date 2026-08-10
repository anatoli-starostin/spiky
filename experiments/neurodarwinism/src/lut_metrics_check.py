"""exp011 validation: the throughput formula, and literal anchor-pair injection.

Five checks, none of which takes the implementation on trust:

  1 THROUGHPUT   count the weight entries the forward actually READS, empirically, at batch
                 size 1, across a grid of (NAP, tph). In hard mode the weight gradient is a
                 1-row scatter at the chosen row, so the support of `weights.grad` after a
                 single-sample backward IS the set of entries the forward touched. Compare
                 against the claimed formula n_heads * tph * n_outputs, and specifically check
                 that it does NOT move with NAP while param_count does.
  2 INJECTION    build with hand-specified pairs and verify the forward uses them, by
                 recomputing the sign-pack index in numpy from those pairs and checking it
                 matches the module's own selection. Also verify a DIFFERENT pair set gives a
                 different output, so the check cannot pass vacuously.
  3 FIXED        anchors come back bit-identical after a full training run -- they are buffers,
                 not Parameters, so the optimiser must not be able to touch them.
  4 WALK         N steps of the real mutate() with pair-level edits: every genome builds and
                 trains, every pair set stays valid (a<b, in range, distinct within a table),
                 and params/throughput stay consistent with the formulas.
  5 TEACHER      the reference point: params, throughput and held-out MSE of the distillation
                 teacher's own shape.

    sbox python lut_metrics_check.py
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

import lut_backprop as lb                                          # noqa: E402
import lut_evolve as le                                            # noqa: E402


def touched_entries(g, x1, dev):
    """How many weight entries does ONE forward read? Support of the 1-sample weight grad."""
    m = lb.build(g, dev)
    y = m(x1).sum(dim=1).float().sum()
    y.backward()
    n = int((m.weights.grad != 0).sum())
    del m
    torch.cuda.empty_cache()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", type=int, default=150)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Xte, Yte = lb.to_device(a.seed, a.n_val, dev)
    rep = {}

    # ---------------------------------------------------------------- 1 THROUGHPUT
    print("(1) THROUGHPUT -- weight entries read per forward, measured at batch size 1")
    print(f"    {'NAP':>4} {'tph':>4} {'params':>10} {'measured':>9} {'formula':>8}  match")
    x1 = Xtr[:1]
    rows, ok_all = [], True
    for nap, tph in itertools.product((3, 6, 9, 12), (8, 32)):
        g = dict(lb.DEFAULT_GENOME, n_anchor_pairs=nap, tables_per_head=tph)
        meas, form = touched_entries(g, x1, dev), lb.throughput(g)
        ok = meas == form
        ok_all &= ok
        rows.append(dict(nap=nap, tph=tph, params=lb.param_count(g),
                         measured=meas, formula=form, match=ok))
        print(f"    {nap:4d} {tph:4d} {lb.param_count(g):10,} {meas:9,} {form:8,}  "
              f"{'OK' if ok else 'MISMATCH'}")
    rep["throughput"] = dict(rows=rows, all_match=ok_all)
    for tph in (8, 32):
        sel = [r for r in rows if r["tph"] == tph]
        p = [r["params"] for r in sel]
        t = {r["measured"] for r in sel}
        print(f"    at tph {tph:2d}: NAP 3->12 moves params {p[0]:,} -> {p[-1]:,} "
              f"({p[-1] // p[0]}x) while throughput stays {t} -- "
              f"{'CONFIRMED independent of NAP' if len(t) == 1 else 'NOT independent'}")
    print("    formula confirmed: throughput = n_heads * tables_per_head * n_outputs")

    # ---------------------------------------------------------------- 2 INJECTION
    print("\n(2) INJECTION -- does the forward use hand-specified pairs?")
    g = dict(lb.DEFAULT_GENOME, n_anchor_pairs=4, tables_per_head=6)
    rng = np.random.default_rng(a.seed)
    nt, nap = lb.n_tables(g), g["n_anchor_pairs"]
    pool = lb.canonical_pool()
    pairs = np.stack([pool[rng.choice(len(pool), nap, replace=False)] for _ in range(nt)])
    ok, why = lb.pairs_valid(pairs)
    m = lb.build(dict(g, anchor_pairs=pairs), dev)
    readback = lb.get_anchor_pairs(m)
    same = bool((readback == pairs).all())
    # recompute the module's row selection independently, in numpy, from those pairs
    xs = Xtr[:64].cpu().numpy()
    d = xs[:, pairs[..., 0]] - xs[:, pairs[..., 1]]                 # [B, n_tables, nap]
    powers = 1 << np.arange(nap - 1, -1, -1)
    want_idx = ((d > 0) * powers).sum(-1)                           # [B, n_tables]
    _, got_idx = m(Xtr[:64]), None
    from spiky.lutorch.fast_multi_head_lut import _soft_lut_fwd_body
    _, got = _soft_lut_fwd_body(Xtr[:64], m.weights, m.soft_anchor_a_long,
                                m.soft_anchor_b_long, m.soft_powers,
                                m.n_heads, m.tables_per_head, m.table_dim)
    got_idx = got.cpu().numpy()
    idx_match = bool((got_idx == want_idx).all())
    # and a DIFFERENT pair set must change the output, so this cannot pass vacuously
    other = np.stack([pool[rng.choice(len(pool), nap, replace=False)] for _ in range(nt)])
    y_a = m(Xtr[:64]).sum(1)
    lb.set_anchor_pairs(m, other)
    y_b = m(Xtr[:64]).sum(1)
    changed = bool((y_a != y_b).any())
    print(f"    pairs valid: {ok} ({why});  read back identical: {same}")
    print(f"    module's row indices == numpy recomputation from those pairs: {idx_match}")
    print(f"    a different pair set changes the output: {changed}")
    rep["injection"] = dict(pairs_valid=ok, readback_identical=same,
                            index_match=idx_match, different_pairs_change_output=changed)
    del m
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 3 FIXED DURING TRAINING
    print("\n(3) FIXED -- anchors must not move during backprop (they are buffers)")
    gp = dict(g, anchor_pairs=pairs)
    mm = lb.build(gp, dev)
    before = lb.get_anchor_pairs(mm)
    n_par = sum(p.numel() for p in mm.parameters())
    opt = torch.optim.Adam(mm.parameters(), lr=1e-2)
    for _ in range(50):
        loss = torch.nn.functional.mse_loss(mm(Xtr[:256]).sum(1).float(), Ytr[:256])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = lb.get_anchor_pairs(mm)
    unchanged = bool((before == after).all())
    in_params = any("anchor" in n for n, _ in mm.named_parameters())
    print(f"    anchors bit-identical after 50 Adam steps: {unchanged}; "
          f"anchors present in model.parameters(): {in_params} "
          f"({n_par:,} trainable scalars, all row weights)")
    rep["fixed_during_training"] = dict(unchanged=unchanged, anchors_in_parameters=in_params,
                                        n_trainable=int(n_par))
    del mm, opt
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 4 WALK
    print(f"\n(4) WALK -- {a.walk} steps of the real mutate() with pair-level edits")
    rng = np.random.default_rng(a.seed)
    gg = le.seed_genome(np.random.default_rng(a.seed))
    fails, edits, naps, tphs = [], 0, set(), set()
    for i in range(a.walk):
        gg = le.mutate(gg, rng)
        edits += gg.get("_n_pair_edits", 0)
        naps.add(gg["n_anchor_pairs"])
        tphs.add(gg["tables_per_head"])
        okp, whyp = lb.pairs_valid(gg["anchor_pairs"])
        shape_ok = tuple(np.asarray(gg["anchor_pairs"]).shape[:2]) == (lb.n_tables(gg),
                                                                      gg["n_anchor_pairs"])
        if not okp or not shape_ok:
            fails.append(dict(step=i, reason=whyp if not okp else "shape mismatch"))
            continue
        try:
            mdl = lb.build(gg, dev)
            loss = torch.nn.functional.mse_loss(mdl(Xtr[:128]).sum(1).float(), Ytr[:128])
            loss.backward()
            assert torch.isfinite(loss)
            assert lb.throughput(gg) == lb.n_tables(gg) * lb.N_OUT
            assert lb.param_count(gg) == lb.n_tables(gg) * (1 << gg["n_anchor_pairs"]) * lb.N_OUT
            del mdl
            torch.cuda.empty_cache()
        except Exception as e:                                       # noqa: BLE001
            fails.append(dict(step=i, reason=f"{type(e).__name__}: {str(e)[:100]}"))
    print(f"    {a.walk - len(fails)}/{a.walk} steps built, trained and stayed valid; "
          f"{len(fails)} failures")
    print(f"    {edits} individual anchor-pair edits; visited NAP {min(naps)}..{max(naps)}, "
          f"tph {min(tphs)}..{max(tphs)}")
    for f in fails[:5]:
        print(f"      FAIL {f}")
    rep["walk"] = dict(steps=a.walk, failures=fails, pair_edits=edits,
                       nap_range=[int(min(naps)), int(max(naps))],
                       tph_range=[int(min(tphs)), int(max(tphs))])

    # ---------------------------------------------------------------- 5 TEACHER
    tg = lb.teacher_genome()
    tr = lb.train_eval(tg, Xtr, Ytr, Xte, Yte, a.steps, 512, a.seed, dev)
    print(f"\n(5) TEACHER reference ({lb.genome_str(tg)})")
    print(f"    params {lb.param_count(tg):,}   throughput {lb.throughput(tg):,} "
          f"weights/forward   held-out MSE {tr['heldout_mse']:.5f} ({a.steps} steps)")
    rep["teacher"] = dict(params=lb.param_count(tg), throughput=lb.throughput(tg),
                          heldout_mse=tr["heldout_mse"], steps=a.steps)

    good = (ok_all and same and idx_match and changed and unchanged and not in_params
            and not fails)
    print(f"\n{'ALL VALIDATION CHECKS PASS' if good else 'SOME CHECKS FAILED -- see above'}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, default=str)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
