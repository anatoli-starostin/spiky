"""exp012: does ANY (gain, inh_coeff) regime make the diagonal TTFS readout work?

Fixed wiring, fixed weights, fixed delays -- only the two global meta-parameters move. For
each setting: how much the hidden layer fires, how much the output first-spike time VARIES
across inputs (zero variance means the output is a constant and can only predict the mean),
and the MSE floor under three readout forms.

The point is to separate "the network is in a bad dynamical regime" from "the readout form
is wrong". If the diagonal floor stays at chance across the whole sweep while the linear
floor is far below it everywhere, the regime is not the problem.
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--n-val", type=int, default=1024)
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
    g0 = pool[int(fin[np.argmin(ewma[fin])])]

    _, _, Xp, Yp, Xv, Yv = load(256, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    Xtr, Ytr, _ = sample_batch(Xp, Yp, a.n_train, a.seed, 11)
    Ttr = T.target_offsets(Ytr)
    Xva, Yva = Xv[:a.n_val], Yv[:a.n_val]
    Tva = T.target_offsets(Yva)
    chance = T.constant_baseline(Yva)

    rows = []
    for gain in (50.0, 100.0, 200.0, 400.0, 800.0):
        for coeff in (0.1, 0.5, 1.0, 2.0):
            g = {**g0, "gain": gain, "inh_coeff": coeff}
            ftr = features(g, Xtr, enc, a.device, chunk=128)
            fva = features(g, Xva, enc, a.device, chunk=128)
            hc = np.column_stack([fva["exc_count"], fva["inh_count"]])
            ow = fva["out_win"]
            at, bt = G.analytic_affine(ftr["out_win"], Ttr)
            diag = float((((at * ow + bt) - Tva) ** 2).mean())
            lin, _ = ridge_fit_score(ftr["out_win"], Ttr, ow, Tva)
            hid, _ = ridge_fit_score(
                np.column_stack([ftr["exc_first"], ftr["inh_first"]]), Ttr,
                np.column_stack([fva["exc_first"], fva["inh_first"]]), Tva)
            rows.append(dict(
                gain=gain, coeff=coeff,
                hidden_spikes_per_neuron=float(hc.mean()),
                frac_hidden_silent=float((hc == 0).mean()),
                out_silent_in_window=float((ow >= T.READOUT_WINDOW).mean()),
                out_sd_across_inputs=float(np.mean([ow[:, d].std() for d in range(T.N_OUT)])),
                out_distinct=int(np.mean([len(np.unique(ow[:, d])) for d in range(T.N_OUT)])),
                mse_diag=diag, mse_linear=lin, mse_hidden=hid))
            r = rows[-1]
            print(f"  gain {gain:6.0f} coeff {coeff:4.1f}   hid {r['hidden_spikes_per_neuron']:5.2f} sp/n  "
                  f"silent {100 * r['frac_hidden_silent']:5.1f}%  out-sd {r['out_sd_across_inputs']:5.2f}  "
                  f"| diag {diag:7.2f}  linear {lin:7.2f}  hidden {hid:7.2f}", flush=True)

    R = dict(chance=chance, target_sd=float(Tva.std()), rows=rows,
             best_diag=min(r["mse_diag"] for r in rows),
             best_linear=min(r["mse_linear"] for r in rows),
             best_hidden=min(r["mse_hidden"] for r in rows))
    print(f"\nchance {chance:.2f} | best diag {R['best_diag']:.2f} | "
          f"best linear {R['best_linear']:.2f} | best hidden {R['best_hidden']:.2f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
