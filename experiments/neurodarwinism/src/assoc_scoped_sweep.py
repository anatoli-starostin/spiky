"""exp010 follow-up: does SCOPED, HOMEOSTATIC STDP write the input->readout association?

The pre-flight left one open question. Teacher-clamped STDP destroyed the reservoir, and at
every rate that left a network standing the true teacher was indistinguishable from a randomly
permuted one. Two candidate causes were never separated:

  SCOPE        all ~96k excitatory synapses were plastic, so the clamp's LTD fell on the whole
               reservoir when the association only needs the readout cells' ~30-100 afferents.
  HOMEOSTASIS  weight_scaling_cf is 0 on the stock metas, so 0 is an absorbing weight and a
               depressed synapse never comes back.

--freeze-reservoir fixes the first, --weight-scaling-cf the second. This sweeps them together
against the one number that matters: does the TRUE-teacher held-out tau beat the
SHUFFLED-teacher tau by more than the member-to-member sd? The pre-flight's best gap was
+0.008 against a sd of 0.03. That is the bar.

Every configuration is run twice from identical seed genomes -- once with the teacher paired to
its own input batch, once with the teacher ticks taken from a permuted batch. The permutation
preserves the clamp's rate, its tick distribution and its STDP load, and destroys only the
pairing, so the difference between the two arms is the association and nothing else.

    sbox python assoc_scoped_sweep.py --out sweep.json
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import steady_state as ss                                              # noqa: E402
from assoc_sanity import evaluate, train, weight_health                # noqa: E402
from harness import LatencyEncoder, N_EXC, N_OUT, N_TICKS              # noqa: E402
from data import load                                                 # noqa: E402


def fanouts(fs):
    return dict(fanout_e=max(1, round(80 / fs)), fanout_i=max(1, round(20 / fs)),
                fanout_inh=max(1, round(100 / fs)), fanout_in=max(1, round(100 / fs)),
                fanin_out=max(1, round(100 / fs)))


def one_arm(cfg, a, Xpool, Ypool, Xval, Yval, enc, dev, shuffle):
    """Build a fresh pool from the same seed genomes, train it, and score it."""
    fs, lr, wsc, freeze = cfg
    ss.ASSOC_FREEZE_RESERVOIR = freeze
    ss.ASSOC_WSC = wsc
    genomes = [ss.seed_genome(np.random.default_rng(a.seed * 100 + i), a.w_max, **fanouts(fs))
               for i in range(a.pool)]
    h = ss.build_pool(genomes, dev, seed=1, stdp_lr=lr, w_max=a.w_max)
    train(h, Xpool, Ypool, enc, a.batch, a.seed, a.train_batches, a.current, a.current,
          shuffle=shuffle, tag="shuffled" if shuffle else "paired")
    ev = evaluate(h, Xval, Yval, enc, a.current, a.current,
                  ("shuffled" if shuffle else "paired"), offset=a.teacher_offset)
    ev["weights"] = weight_health(h, genomes)
    del h
    torch.cuda.empty_cache()
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--train-batches", type=int, default=400)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-offset", type=int, default=32)
    ap.add_argument("--teacher-levels", type=int, default=32)
    ap.add_argument("--fanout-scales", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--stdp-lrs", type=float, nargs="+", default=[0.001, 0.01])
    ap.add_argument("--wscs", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1])
    ap.add_argument("--no-freeze-control", action="store_true",
                    help="skip the unfrozen control row per fan-out")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ss.ASSOC = True
    ss.ASSOC_TEACHER_OFFSET = a.teacher_offset
    ss.ASSOC_TEACHER_CURRENT = a.current
    X, Y, Xpool, Ypool, Xval, Yval = load(a.batch, a.seed, a.n_val)
    enc = LatencyEncoder(Xpool)
    ss.fit_target_stats(Ypool, 2.5, a.teacher_levels)

    # Every configuration is scored on the windowed readout [offset, 96) -- bookkeeping fix B.
    # The teacher-ON ceiling for that window was measured at +0.523 in the pre-flight, against
    # a design ceiling (the teacher ticks themselves) of +0.540.
    cfgs = [(fs, lr, wsc, True) for fs, lr, wsc
            in itertools.product(a.fanout_scales, a.stdp_lrs, a.wscs)]
    if not a.no_freeze_control:
        # the pre-flight's regime, re-run here so the comparison is like-for-like
        cfgs += [(fs, lr, 0.0, False) for fs, lr in
                 itertools.product(a.fanout_scales, a.stdp_lrs)]

    print(f"assoc scoped sweep: K={a.pool}, {a.train_batches} clamped batches, "
          f"{a.n_val} held-out states, teacher offset {a.teacher_offset} "
          f"(readout window [{a.teacher_offset}, {N_TICKS})), dev {dev}")
    print(f"{len(cfgs)} configurations x 2 arms (paired / shuffled)\n")

    rows, t0 = [], time.time()
    for i, cfg in enumerate(cfgs):
        fs, lr, wsc, freeze = cfg
        tag = (f"fanout/{fs:g}  lr {lr:g}  wsc {wsc:g}  "
               f"{'FROZEN reservoir' if freeze else 'unfrozen (control)'}")
        print(f"[{i + 1}/{len(cfgs)}] {tag}", flush=True)
        paired = one_arm(cfg, a, Xpool, Ypool, Xval, Yval, enc, dev, shuffle=False)
        shuf = one_arm(cfg, a, Xpool, Ypool, Xval, Yval, enc, dev, shuffle=True)
        gap = paired["corrected_window_mean"] - shuf["corrected_window_mean"]
        sd = max(paired["corrected_sd"], shuf["corrected_sd"])
        rows.append(dict(fanout_scale=fs, stdp_lr=lr, wsc=wsc, freeze=freeze,
                         paired=paired, shuffled=shuf, gap=gap, sd=sd,
                         beats_sd=bool(gap > sd)))
        ro = paired["weights"]["readout_afferents"]
        print(f"      -> gap {gap:+.4f}  (sd {sd:.4f})  "
              f"{'BEATS SD' if gap > sd else 'within noise'}   "
              f"readout-afferent w mean {ro.get('mean', -1):.2f} "
              f"frac<0.1 {ro.get('frac_below_0p1', -1):.3f} "
              f"at-ceiling {ro.get('frac_at_ceiling', -1):.3f}\n", flush=True)

    print(f"\n{'fanout':>7} {'lr':>7} {'wsc':>7} {'frozen':>7} | {'sp/cell':>8} "
          f"{'paired':>8} {'shuffled':>9} {'gap':>8} {'sd':>7}  verdict")
    for r in rows:
        print(f"{r['fanout_scale']:7g} {r['stdp_lr']:7g} {r['wsc']:7g} "
              f"{str(r['freeze']):>7} | {r['paired']['spikes_per_cell']:8.2f} "
              f"{r['paired']['corrected_window_mean']:+8.4f} "
              f"{r['shuffled']['corrected_window_mean']:+9.4f} "
              f"{r['gap']:+8.4f} {r['sd']:7.4f}  "
              f"{'BEATS SD' if r['beats_sd'] else '-'}")
    print(f"\n{time.time() - t0:.0f}s total")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
