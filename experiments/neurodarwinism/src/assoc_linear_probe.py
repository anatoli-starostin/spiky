"""exp010 follow-up control: is there anything in the reservoir for STDP to FIND?

Both the pre-flight and the scoped sweep answer "did teacher-clamped STDP write the
association?". Neither answers the prior question: *could any rule have written it?* If the
frozen reservoir's response to a state carries no information about that state's action vector,
then no learning rule on the readout cells' afferents can produce one, and the negative result
is about the substrate rather than about STDP.

So: freeze the reservoir, read every excitatory neuron's first-spike tick and spike count as a
feature vector, fit a RIDGE REGRESSION from those features to the actions on training states,
and score it on held-out states with the chapter's corrected tau-b. Ridge is the generous
control -- it is the best linear readout of the whole reservoir, given the labels directly and
in closed form, and it sees 800 neurons where the 6 readout cells see only their own ~30-100
afferents. It is an upper bound on what any local rule could extract, not a competitor.

Three readouts are reported, in decreasing generosity:
  ALL EXC       ridge on all 800 excitatory neurons -- what the reservoir knows
  AFFERENTS     ridge restricted to the presynaptic partners the 6 readout cells actually have
                -- what a perfect rule on THIS wiring could reach
  READOUT ONLY  the 6 readout cells' own first spikes, untrained -- where we currently are
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import steady_state as ss                                                    # noqa: E402
from harness import (LatencyEncoder, N_EXC, N_OUT, N_TICKS,                  # noqa: E402
                     kendall_tau_b, run_episode)
from data import load                                                        # noqa: E402


def corrected(pred, Y, n_shuf=40):
    raw = float(kendall_tau_b(pred, Y).mean())
    nl = float(np.mean([
        kendall_tau_b(pred, Y[np.random.default_rng(k).permutation(Y.shape[0])]).mean()
        for k in range(n_shuf)]))
    return raw, nl, raw - nl


def features(h, X, enc, current, chunk=128):
    """[N, 2*N_EXC]: every excitatory neuron's first-spike tick and its spike count.

    Both, because a cell that fires late-and-often and a cell that fires once carry different
    information and first-spike alone throws the second away.
    """
    from spiky.spnet.spnet import NeuronDataType
    sp, dev = h["spnet"], h["device"]
    ids = torch.tensor(h["ids"][0], dtype=torch.int32, device=dev)
    n = ids.numel()
    out = []
    for i in range(0, X.shape[0], chunk):
        Xb = X[i:i + chunk]
        run_episode(h, Xb, enc, current)
        R = sp.export_neuron_data(ids, Xb.shape[0], NeuronDataType.Spike,
                                  0, N_TICKS - 1).cpu().numpy().reshape(Xb.shape[0], n, N_TICKS)
        has = R.any(-1)
        first = np.where(has, R.argmax(-1), N_TICKS).astype(np.float32)
        out.append(np.concatenate([first, R.sum(-1).astype(np.float32)], 1))
    return np.concatenate(out, 0)


def ridge_tau(Ftr, Ytr, Fte, Yte, lam=10.0):
    """Closed-form ridge on standardised features, scored with corrected tau-b.

    Predicted VALUE is the regression output directly (we regress onto the raw actions), so no
    sign flip: the latency convention only applies to spike times, and these are actions.
    """
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    A = np.concatenate([(Ftr - mu) / sd, np.ones((Ftr.shape[0], 1), np.float32)], 1)
    B = np.concatenate([(Fte - mu) / sd, np.ones((Fte.shape[0], 1), np.float32)], 1)
    G = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
    W = np.linalg.solve(G, A.T @ Ytr)
    pred_tr, pred_te = A @ W, B @ W
    return dict(train=corrected(pred_tr, Ytr)[2], heldout=corrected(pred_te, Yte)[2],
                heldout_raw=corrected(pred_te, Yte)[0], n_features=int(Ftr.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--fanout-scale", type=float, default=3.0)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ss.ASSOC = True
    X, Y, Xpool, Ypool, Xval, Yval = load(64, a.seed, a.n_val)
    enc = LatencyEncoder(Xpool)
    ss.fit_target_stats(Ypool, 2.5, 32)
    fs = a.fanout_scale
    fan = dict(fanout_e=max(1, round(80 / fs)), fanout_i=max(1, round(20 / fs)),
               fanout_inh=max(1, round(100 / fs)), fanout_in=max(1, round(100 / fs)),
               fanin_out=max(1, round(100 / fs)))
    g = ss.seed_genome(np.random.default_rng(a.seed), a.w_max, **fan)
    h = ss.build_pool([g], dev, seed=1, stdp_lr=0.0, w_max=a.w_max)
    print(f"linear probe: fan-out /{fs:g}, {g['weight'].size:,} synapses, "
          f"{a.n_train} train / {a.n_val} held-out states, dev {dev}")

    idx = np.random.default_rng(a.seed + 99).choice(Xpool.shape[0], a.n_train, replace=False)
    Ftr = features(h, Xpool[idx], enc, a.current)
    Fte = features(h, Xval, enc, a.current)
    Ytr, Yte = Ypool[idx], Yval
    print(f"features {Ftr.shape} (first-spike tick + spike count per excitatory neuron)")

    # which reservoir neurons actually project to a readout cell on THIS genome
    aff = np.unique(g["src_idx"][(g["src_pool"] == ss.EXC) & (g["tgt_pool"] == ss.EXC)
                                 & (g["tgt_idx"] >= N_EXC - N_OUT)])
    ro = np.arange(N_EXC - N_OUT, N_EXC)
    # (cells, use_count). The last row is the one directly comparable to the TTFS metric: the
    # SAME six numbers the readout reads, decoded linearly instead of by their rank order. The
    # gap between it and the measured TTFS tau is information the readout code throws away
    # rather than information the network lacks.
    sets = {"ALL EXC (800 cells)": (np.arange(N_EXC), True),
            f"AFFERENTS of readout ({aff.size} cells)": (aff, True),
            "READOUT CELLS only (6), spike+count": (ro, True),
            "READOUT CELLS only (6), FIRST SPIKE only": (ro, False)}
    # THE APPLES-TO-APPLES BASELINE: the chapter's actual metric, on the same held-out states
    # and the same six numbers the last probe row decodes. pred = -first, because earlier spike
    # means larger action. Any gap between this and that row is signal the RANK-ORDER READOUT
    # discards, not signal the network lacks.
    raw_ttfs, null_ttfs, cor_ttfs = corrected(-Fte[:, ro], Yte)
    print(f"  {'TTFS RANK ORDER of the same 6 (the metric)':50s} held-out corrected tau "
          f"{cor_ttfs:+.4f}  (raw {raw_ttfs:+.4f}, null {null_ttfs:+.4f})")
    rep = dict(config=vars(a), n_afferent_sources=int(aff.size),
               ttfs_rank_order=dict(raw=raw_ttfs, null=null_ttfs, corrected=cor_ttfs),
               results={})
    for name, (sel, use_count) in sets.items():
        cols = np.concatenate([sel, sel + N_EXC]) if use_count else sel
        best = max((ridge_tau(Ftr[:, cols], Ytr, Fte[:, cols], Yte, lam) | dict(lam=lam)
                    for lam in a.lams), key=lambda r: r["heldout"])
        rep["results"][name] = best
        print(f"  {name:34s} held-out corrected tau {best['heldout']:+.4f} "
              f"(train {best['train']:+.4f}, lambda {best['lam']:g}, "
              f"{best['n_features']} features)")

    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
