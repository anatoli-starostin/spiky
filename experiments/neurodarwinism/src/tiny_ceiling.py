"""exp012: is the substrate ABLE to represent the target, or is the search the problem?

One forward pass per network, from which everything else is derived:

  * out_win     the readout the run actually uses: first spike inside [64,96), rebased 0..31,
                silence = 32
  * out_full    first spike anywhere in the 96-tick episode, silence = 96. If this carries
                information the windowed readout does not, the WINDOW is throwing it away.
  * out_count   spikes per output neuron per episode -- a rate code, not a timing code
  * hid_first   first spike per hidden neuron over the full episode, silence = 96
  * hid_count   spikes per hidden neuron

Then fit the best readout each feature set allows, by ridge regression on a TRAINING batch and
scored on held-out, so every number is honest:

  (a) the evolved affine                    what the run actually achieves
  (b) free per-output affine on out_win     the "affine ceiling", fitted properly
  (c) linear on out_full / out_count        does the window or the code discard signal?
  (d) linear on ALL hidden features         is the information in the network at all?

If (d) is near chance, the substrate cannot represent the target and no amount of evolution
will help. If (d) is far below chance, the information is there and the readout or the search
is losing it.
"""
import argparse
import json

import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder


def features(g, X, enc, device="cuda", chunk=256):
    """One forward pass -> every timing/rate feature, for hidden and output neurons."""
    from spiky.spnet.spnet import NeuronDataType
    H = G.build([g], device=device)
    sp, ids = H["spnet"], H["ids"]
    cols = ids[2]
    outs = []
    for lo in range(0, X.shape[0], chunk):
        Xc = X[lo:lo + chunk]
        B = Xc.shape[0]
        tk = enc(Xc)
        va = np.zeros((B, T.T_IN, cols.size), np.float32)
        for b in range(B):
            for j in range(T.N_IN):
                va[b, tk[b, j], j::T.N_IN] = 200.0
        sp.process_ticks(n_ticks_to_process=T.N_TICKS, batch_size=B, n_input_ticks=T.T_IN,
                         input_values=torch.as_tensor(va, device=device),
                         sparse_input=T._sparse_ids(cols, B, T.T_IN, device),
                         do_train=False, do_record_voltage=False, do_reset_context=True,
                         _stdp_period=32)
        got = {}
        for k, nm in ((0, "exc"), (1, "inh"), (3, "out")):
            oid = torch.as_tensor(np.ascontiguousarray(ids[k], dtype=np.int32), device=device)
            R = sp.export_neuron_data(oid, B, NeuronDataType.Spike, 0, T.N_TICKS - 1)
            fire = R.ne(0)                                     # [B, n, 96]
            W = T.READOUT_WINDOW
            wgt = torch.arange(T.N_TICKS, 0, -1, device=R.device, dtype=R.dtype)
            first_full = (T.N_TICKS - (fire * wgt).amax(-1)).double().cpu().numpy()
            got[nm + "_first"] = first_full
            got[nm + "_count"] = fire.sum(-1).double().cpu().numpy()
            if nm == "out":
                w2 = torch.arange(W, 0, -1, device=R.device, dtype=R.dtype)
                got["out_win"] = (W - (fire[..., T.N_TICKS - W:] * w2).amax(-1)
                                  ).double().cpu().numpy()
        outs.append(got)
    return {k: np.concatenate([o[k] for o in outs], 0) for k in outs[0]}


def ridge_fit_score(Ftr, Ttr, Fva, Tva, lam=1.0):
    """Least squares with a small ridge, fitted on TRAINING and scored on HELD-OUT."""
    A = np.column_stack([Ftr, np.ones(len(Ftr))])
    B = np.column_stack([Fva, np.ones(len(Fva))])
    n = A.shape[1]
    W = np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ Ttr)
    return float(((B @ W - Tva) ** 2).mean()), W


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
        G.set_delay_levels(list(range(1, 64, 2)) if a.delay_levels == "odd"
                           else [int(x) for x in a.delay_levels.split(",")])
    G.QUANTIZED = a.quantized

    from tiny_grow_evolve import load_ckpt
    pool, ewma, age, rnd, hist, best, _ = load_ckpt(a.ckpt)
    fin = np.where(np.isfinite(ewma))[0]
    g = pool[int(fin[np.argmin(ewma[fin])])]

    _, _, Xp, Yp, Xv, Yv = load(256, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    Xtr, Ytr, _ = sample_batch(Xp, Yp, a.n_train, a.seed, 11)
    Ttr = T.target_offsets(Ytr)
    Tva = T.target_offsets(Yv)
    chance = T.constant_baseline(Yv)

    ftr = features(g, Xtr, enc, a.device)
    fva = features(g, Xv, enc, a.device)

    R = dict(label=a.label, ckpt=a.ckpt, round=rnd, chance=chance,
             n_train=a.n_train, n_val=int(Xv.shape[0]),
             genome=dict(n_syn=int(g["mask"].sum()), n_active=G.n_active(g),
                         gain=G.gain_of(g), inh_coeff=G.inh_coeff_of(g)))

    # ---- (a) the evolved readout, exactly as the run scores it
    aa, bb = G.affine_of(g)
    R["a_evolved_affine"] = float((((aa * fva["out_win"] + bb) - Tva) ** 2).mean())
    R["a_raw_no_affine"] = float(((fva["out_win"] - Tva) ** 2).mean())

    # ---- (b)-(d) the best a linear readout can do from each feature set
    sets = {
        "b_out_win (6 feats, the run's own readout)": ("out_win",),
        "c1_out_full 96-tick first spike (6)": ("out_first",),
        "c2_out_count spike RATE (6)": ("out_count",),
        "c3_out_full + out_count (12)": ("out_first", "out_count"),
        "d1_hidden first spikes (50)": ("exc_first", "inh_first"),
        "d2_hidden counts (50)": ("exc_count", "inh_count"),
        "d3_hidden first + counts (100)": ("exc_first", "inh_first",
                                           "exc_count", "inh_count"),
        "d4_EVERYTHING hidden + out (112)": ("exc_first", "inh_first", "exc_count",
                                             "inh_count", "out_first", "out_count"),
    }
    R["linear_ceilings"] = {}
    for nm, keys in sets.items():
        Ftr = np.column_stack([ftr[k] for k in keys])
        Fva = np.column_stack([fva[k] for k in keys])
        mse, _ = ridge_fit_score(Ftr, Ttr, Fva, Tva)
        R["linear_ceilings"][nm] = dict(n_features=Ftr.shape[1], heldout_mse=mse,
                                        pct_of_chance=100 * mse / chance)

    # ---- (2) OUTPUT BEHAVIOUR
    ow, of_, oc = fva["out_win"], fva["out_first"], fva["out_count"]
    R["output_behaviour"] = dict(
        frac_no_spike_in_window=float((ow >= T.READOUT_WINDOW).mean()),
        frac_no_spike_at_all=float((of_ >= T.N_TICKS).mean()),
        win_sd_across_inputs=[float(ow[:, d].std()) for d in range(T.N_OUT)],
        full_sd_across_inputs=[float(of_[:, d].std()) for d in range(T.N_OUT)],
        count_mean=[float(oc[:, d].mean()) for d in range(T.N_OUT)],
        count_sd_across_inputs=[float(oc[:, d].std()) for d in range(T.N_OUT)],
        win_distinct=[int(len(np.unique(ow[:, d]))) for d in range(T.N_OUT)],
        full_first_min=float(of_.min()), full_first_max=float(of_.max()),
        target_sd=float(Tva.std()))

    # ---- (3) HIDDEN REGIME
    hf = np.column_stack([fva["exc_first"], fva["inh_first"]])
    hc = np.column_stack([fva["exc_count"], fva["inh_count"]])
    R["hidden_regime"] = dict(
        mean_spikes_per_neuron=float(hc.mean()),
        frac_neurons_always_silent=float((hc.sum(0) == 0).mean()),
        frac_episodes_silent=float((hc == 0).mean()),
        frac_first_at_tick_0=float((hf == 0).mean()),
        first_spike_mean=float(hf[hf < T.N_TICKS].mean()) if (hf < T.N_TICKS).any() else None,
        first_spike_sd_across_inputs=float(np.mean([hf[:, i].std() for i in range(hf.shape[1])])),
        count_sd_across_inputs=float(np.mean([hc[:, i].std() for i in range(hc.shape[1])])),
        max_spikes_seen=float(hc.max()))

    # ---- (4) INPUT SEPARATION: does the hidden state depend on the input at all?
    ranks = []
    for i in range(hf.shape[1]):
        if hf[:, i].std() > 1e-9:
            r = [abs(np.corrcoef(Xv[:, j], hf[:, i])[0, 1]) for j in range(T.N_IN)]
            ranks.append(max(x for x in r if np.isfinite(x)))
    R["input_separation"] = dict(
        n_hidden_with_variance=len(ranks),
        max_abs_corr_input_to_hidden_first=float(np.max(ranks)) if ranks else 0.0,
        mean_abs_corr=float(np.mean(ranks)) if ranks else 0.0,
        n_distinct_hidden_patterns=int(len(np.unique(hf, axis=0))),
        n_inputs=int(Xv.shape[0]))

    print(json.dumps(T.jsonable(R), indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
