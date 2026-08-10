"""exp010 follow-up: is there ANY operating point where the assoc readout is TTFS-codeable?

The first sanity pass found the readout cells firing ~11 times per 96-tick episode and the
whole reservoir depressing to silence within ~64 STDP batches. Both are properties of the
reservoir's gain and the STDP rate, not of the associative idea, so before calling the idea
dead we sweep the two knobs that control them:

  --fanout-scale   divides every seed fan-out, so it sets the recurrent gain
  --stdp-lr        sets how fast LTD can run the weights down

The target regime is roughly 1 spike per readout cell per episode (so "first spike" is a
code, not a sample of an ongoing train) that SURVIVES training (so there is still a net
after STDP has run).
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

import steady_state as ss                                              # noqa: E402
from harness import LatencyEncoder, N_OUT, N_TICKS, T_IN, run_episode  # noqa: E402
from data import load, sample_batch                                   # noqa: E402


def activity(h, Xv, Yv, enc, current):
    _, R = run_episode(h, Xv, enc, current)
    rate = float(R.sum(-1).mean())
    has = R.any(-1)
    f = np.where(has, R.argmax(-1), N_TICKS)
    return dict(spikes_per_cell=rate,
                silent_frac=float((~has).mean()),
                frac_input_phase=float((f < T_IN).mean()),
                frac_readout_phase=float(((f >= 2 * T_IN) & (f < N_TICKS)).mean()),
                spikes_in_readout_phase=float(R[..., 2 * T_IN:].sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fanout-scales", type=float, nargs="+", default=[1.0, 4.0, 10.0, 20.0])
    ap.add_argument("--stdp-lrs", type=float, nargs="+", default=[0.01, 0.001])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ss.ASSOC = True
    ss.ASSOC_TEACHER_OFFSET = 64
    ss.ASSOC_TEACHER_CURRENT = a.current
    X, Y, Xpool, Ypool, Xval, Yval = load(a.batch, a.seed, a.n_val)
    enc = LatencyEncoder(Xpool)
    ss.fit_target_stats(Ypool, 2.5, 32)

    print(f"{'fanout':>7} {'lr':>7} | {'cold sp/cell':>12} {'in-phase':>9} {'out-phase':>10} "
          f"| {'trained sp/cell':>15} {'silent':>7} {'out-phase sp':>12}")
    rows = []
    for fs, lr in itertools.product(a.fanout_scales, a.stdp_lrs):
        fan = dict(fanout_e=max(1, round(80 / fs)), fanout_i=max(1, round(20 / fs)),
                   fanout_inh=max(1, round(100 / fs)), fanout_in=max(1, round(100 / fs)),
                   fanin_out=max(1, round(100 / fs)))
        genomes = [ss.seed_genome(np.random.default_rng(a.seed * 100 + i), a.w_max, **fan)
                   for i in range(a.pool)]
        h = ss.build_pool(genomes, dev, seed=1, stdp_lr=lr, w_max=a.w_max)
        cold = activity(h, Xval, Yval, enc, a.current)
        for i in range(a.batches):
            Xb, Yb, _ = sample_batch(Xpool, Ypool, a.batch, a.seed, 900_000 + i)
            run_episode(h, Xb, enc, a.current, train=True,
                        teacher_ticks=ss.teacher_ticks_for(Yb), teacher_current=a.current)
        warm = activity(h, Xval, Yval, enc, a.current)
        rows.append(dict(fanout_scale=fs, stdp_lr=lr, cold=cold, trained=warm,
                         n_syn=int(genomes[0]["weight"].size)))
        print(f"{fs:7.1f} {lr:7.4f} | {cold['spikes_per_cell']:12.2f} "
              f"{cold['frac_input_phase']:9.2f} {cold['frac_readout_phase']:10.2f} "
              f"| {warm['spikes_per_cell']:15.2f} {warm['silent_frac']:7.2f} "
              f"{warm['spikes_in_readout_phase']:12.2f}", flush=True)
        del h
        torch.cuda.empty_cache()

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
