"""exp012: split the readout gap into 'the search failed' vs 'the readout FORM is too weak'.

The affine genes can only express a DIAGONAL map -- y_d = a_d * out_d + b_d, each output
calibrated against itself. My first ceiling pass fitted a full 6x6 linear map, which is a
strictly richer object, so its MSE cannot be attributed to the genes. This separates them:

    evolved diagonal     what the run actually found
    LS diagonal          the best the genes COULD have expressed   -> gap = search failure
    LS full 6x6          a richer readout form                     -> gap = form limitation
    LS hidden            what the network actually knows           -> gap = output bottleneck

Fitted on a TRAINING batch, scored on held-out, throughout.
"""
import argparse
import json

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder
from tiny_ceiling import features, ridge_fit_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="net")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--weight-levels", default=None)
    ap.add_argument("--delay-levels", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.weight_levels:
        G.set_weight_levels([float(x) for x in a.weight_levels.split(",")])
    if a.delay_levels:
        G.set_delay_levels(list(range(1, 64, 2)))
    G.QUANTIZED = a.quantized

    from tiny_grow_evolve import load_ckpt
    pool, ewma, *_ = load_ckpt(a.ckpt)
    fin = np.where(np.isfinite(ewma))[0]
    g = pool[int(fin[np.argmin(ewma[fin])])]

    _, _, Xp, Yp, Xv, Yv = load(256, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    Xtr, Ytr, _ = sample_batch(Xp, Yp, a.n_train, a.seed, 11)
    Ttr, Tva = T.target_offsets(Ytr), T.target_offsets(Yv)
    chance = T.constant_baseline(Yv)

    ftr = features(g, Xtr, enc, a.device)
    fva = features(g, Xv, enc, a.device)
    aa, bb = G.affine_of(g)

    def mse(y):
        return float(((y - Tva) ** 2).mean())

    R = dict(label=a.label, chance=chance)
    R["evolved_diagonal"] = mse(aa * fva["out_win"] + bb)
    R["no_readout_raw"] = mse(fva["out_win"])

    # the best DIAGONAL fit -- exactly the form the 12 genes can express
    at, bt = G.analytic_affine(ftr["out_win"], Ttr)
    R["LS_diagonal_on_out_win"] = mse(at * fva["out_win"] + bt)
    R["LS_diagonal_coeffs"] = dict(a=at.tolist(), b=bt.tolist())
    R["evolved_coeffs"] = dict(a=aa.tolist(), b=bb.tolist())

    # richer forms, for contrast
    for nm, keys in (("LS_full6x6_on_out_win", ("out_win",)),
                     ("LS_full_on_out_full_96tick", ("out_first",)),
                     ("LS_on_out_full_plus_count", ("out_first", "out_count")),
                     ("LS_on_hidden_first", ("exc_first", "inh_first")),
                     ("LS_on_everything", ("exc_first", "inh_first", "exc_count",
                                           "inh_count", "out_first", "out_count"))):
        Ftr = np.column_stack([ftr[k] for k in keys])
        Fva = np.column_stack([fva[k] for k in keys])
        R[nm], _ = ridge_fit_score(Ftr, Ttr, Fva, Tva)

    R["gaps"] = dict(
        search_failure_evolved_minus_LSdiag=R["evolved_diagonal"] - R["LS_diagonal_on_out_win"],
        form_limit_LSdiag_minus_LSfull=R["LS_diagonal_on_out_win"] - R["LS_full6x6_on_out_win"],
        window_loss_LSfull_minus_LSfull96=R["LS_full6x6_on_out_win"]
        - R["LS_full_on_out_full_96tick"],
        output_bottleneck_LSfull96_minus_hidden=R["LS_full_on_out_full_96tick"]
        - R["LS_on_hidden_first"])
    print(json.dumps(T.jsonable(R), indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)


if __name__ == "__main__":
    main()
