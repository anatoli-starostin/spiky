"""exp012: where does the 0.13 floor on the single comparison bit come from?

Four separable suspects, one check each:

  READOUT   given the spikes the champion actually emits, what is the best ANY decode of the
            output's first-spike tick could do? A per-tick lookup (the mean target among
            samples landing on that tick) is that optimum, and it needs no fitting beyond a
            table. If it equals the fitted scale+shift, the readout is not the limit.
  NETWORK   is the bit present in the HIDDEN layer but lost on the way out? Ridge on the 8
            hidden first-spike times answers that.
  WIRING    are the two relevant inputs even connected to the output within the delay budget?
            An unreachable input is a trivial, and fixable, explanation.
  ENCODING  is the sign preserved at all -- is the encoder monotone, and does it ever tie?
"""
import argparse
import json

import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

IA, IB = 0, 16


def setup():
    G.set_hidden_capacity(8, 0)
    G.set_out_per_target(1, "mean")
    G.set_weight_levels([round(0.1 * i, 10) for i in range(11)])
    G.set_delay_levels(list(range(1, 64, 2)))
    G.QUANTIZED = True
    G.MAX_EPISODE_BATCH = 512
    G.set_bit_task(IA, IB)


def per_tick_lut(ticks_t, y_t, ticks_v, y_v, n=33):
    """The MSE-optimal decode of a first-spike tick: the per-tick mean, fitted on TRAIN."""
    glob = float(y_t.mean())
    tab = np.array([y_t[ticks_t == k].mean() if (ticks_t == k).any() else glob
                    for k in range(n)])
    pv = tab[np.clip(ticks_v, 0, n - 1).astype(int)]
    return float(((pv - y_v) ** 2).mean()), tab, pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setup()

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, 12345)
    yt = G.task_targets(Yb, Xb).ravel()
    yv = G.task_targets(Yv, Xv).ravel()
    chance = float(yv.var())
    R = dict(chance=chance, p1_val=float(yv.mean()))

    # ---------------------------------------------------------------- 4 ENCODING sanity
    et, ev = enc(Xb), enc(Xv)
    xa, xb_ = Xv[:, IA], Xv[:, IB]
    ta, tb = ev[:, IA], ev[:, IB]
    ord_ok = ((xa > xb_) == (ta < tb))                 # larger x must fire EARLIER
    ties = (ta == tb)
    R["encoding"] = dict(
        monotone_agreement=float(ord_ok.mean()), tie_fraction=float(ties.mean()),
        agreement_when_not_tied=float(ord_ok[~ties].mean()) if (~ties).any() else 1.0,
        n_distinct_ticks_a=int(len(np.unique(ta))), n_distinct_ticks_b=int(len(np.unique(tb))))
    print(f"4 ENCODING  spike-order agrees with the sign on {100 * ord_ok.mean():.2f}% of "
          f"held-out samples")
    print(f"            ties (both inputs on the SAME tick): {100 * ties.mean():.2f}%   "
          f"-> the bit is UNRECOVERABLE from order on those")
    print(f"            agreement among non-tied: "
          f"{100 * R['encoding']['agreement_when_not_tied']:.2f}%")

    # ---------------------------------------------------------------- load the champion
    from tiny_grow_evolve import load_ckpt
    pool, ewma, *_ = load_ckpt(a.ckpt)
    fin = np.where(np.isfinite(ewma))[0]
    g = pool[int(fin[np.argmin(ewma[fin])])]

    # ---------------------------------------------------------------- 3 WIRING sanity
    m = g["mask"]
    in_out_direct = bool(m[IA, G.C_OUT].any() or m[IB, G.C_OUT].any())   # illegal by design
    reach = {}
    for nm, i in (("x%d" % IA, IA), ("x%d" % IB, IB)):
        tgts = np.where(m[i, G.C_EXC])[0]
        paths = []
        for e in tgts:
            row = G.N_IN + e
            for o in np.where(m[row, G.C_OUT])[0]:
                d = int(g["delay"][i, e]) + int(g["delay"][row, G.N_EXC_MAX + o])
                paths.append(d)
        reach[nm] = dict(n_hidden_targets=int(len(tgts)), n_two_hop_paths=int(len(paths)),
                         delays=sorted(paths)[:12],
                         # an input spike at tick s lands at s + d; the readout window is
                         # [64, 96), and input ticks span 0..31
                         n_paths_landing_in_window=int(sum(64 - 31 <= d < 96 for d in paths)))
        print(f"3 WIRING    {nm}: {len(tgts)} hidden targets, {len(paths)} two-hop paths to "
              f"the output, {reach[nm]['n_paths_landing_in_window']} of them can land in the "
              f"readout window")
    R["wiring"] = dict(reach=reach, direct_input_to_output=in_out_direct,
                       n_synapses=int(m.sum()))

    # ---------------------------------------------------------------- 1/2 the spikes
    H = G.build([g], device="cuda")
    st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
    sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls",
                 readout_map=st["readout_map"])
    diag = float(sv["mse"][0])
    tick_t = st["first"][:, 0, 0].astype(int)
    tick_v = sv["first"][:, 0, 0].astype(int)
    hid_t = st["raw_neurons"][:, 0, :] if st["raw_neurons"] is not None else None
    R["diagls_heldout"] = diag
    R["diagls_accuracy"] = float(((sv["calibrated"][:, 0, 0] > 0.5) == (yv > 0.5)).mean())
    print(f"\n2 READOUT   the run's own diagls readout: held-out {diag:.4f}  "
          f"acc {100 * R['diagls_accuracy']:.2f}%   (chance {chance:.4f})")

    lut_mse, tab, pv = per_tick_lut(tick_t, yt, tick_v, yv)
    R["best_tick_decode"] = lut_mse
    R["best_tick_decode_accuracy"] = float(((pv > 0.5) == (yv > 0.5)).mean())
    R["n_distinct_output_ticks"] = int(len(np.unique(tick_v)))
    print(f"            the BEST possible decode of that same tick (per-tick mean): "
          f"{lut_mse:.4f}  acc {100 * R['best_tick_decode_accuracy']:.2f}%")
    print(f"            output uses {R['n_distinct_output_ticks']} distinct ticks on held-out")

    # is the bit in the hidden layer at all?
    def ridge(F, y, Fv, yv_, lam=1.0):
        A = np.c_[F, np.ones(len(F))]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        p = np.c_[Fv, np.ones(len(Fv))] @ W
        return float(((p - yv_) ** 2).mean()), p

    if hid_t is not None:
        hv = sv["raw_neurons"][:, 0, :]
        # raw_neurons is the OUTPUT group; the hidden layer is not exported, so use what we
        # have plus the encoded input as the upper reference
        hm, hp = ridge(hid_t, yt, hv, yv)
        R["ridge_on_output_neurons"] = hm
    em, ep = ridge(et.astype(float), yt, ev.astype(float), yv)
    R["ridge_on_encoded_input"] = em
    R["ridge_on_encoded_input_accuracy"] = float(((ep > 0.5) == (yv > 0.5)).mean())
    print(f"            ridge on the 17 ENCODED INPUT ticks: {em:.4f}  "
          f"acc {100 * R['ridge_on_encoded_input_accuracy']:.2f}%  "
          f"<- what a linear decode of the input alone gets")

    del H, st, sv
    torch.cuda.empty_cache()

    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
