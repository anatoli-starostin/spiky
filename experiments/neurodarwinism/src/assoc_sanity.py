"""exp010 pre-flight: does the auto-associative readout do anything before we spend 300 rounds?

Four questions, in the order they can fail:

  (1) CLAMP   with the teacher current on, do the 6 readout neurons actually fire at their
              encoded ticks? If the clamp misses, STDP is being taught noise.
  (2) ALIVE   with the teacher off, do they still fire at all, and at more than one distinct
              tick? A silent or fully-tied readout has no order to score.
  (3) SIGNAL  does held-out corrected tau move above chance after teacher-clamped STDP, and
              is it better than the same net before training?
  (4) TRIVIAL is whatever signal exists an actual recall, or just the input volley washing
              through 6 ordinary reservoir cells? Three controls, below.

THE CONTROLS FOR (4), which is the question that decides whether exp010 means anything:
  * COLD          the same net, evaluated before any teacher-clamped STDP. Any tau here is
                  structural -- it is what the random reservoir gives you for free.
  * SHUFFLED      train with the teacher ticks drawn from a PERMUTED batch of actions, so the
                  input->target association is destroyed while every other statistic of the
                  clamp (its rate, its tick distribution, its STDP load) is preserved. If the
                  trained tau survives this, it was never an association.
  * TEACHER-ON    evaluate WITH the clamp still applied. This must be near +1: it measures
                  the clamp, not the network, and is here only to prove the readout path and
                  the metric are wired up the way we think.

    sbox python assoc_sanity.py --teacher-offset 64
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import steady_state as ss                                          # noqa: E402
from harness import (D_MAX, LatencyEncoder, N_EXC, N_OUT, N_TICKS, T_IN,   # noqa: E402
                     kendall_tau_b, run_episode)
from data import load, sample_batch                                # noqa: E402


def corrected_tau(first, Y, p, n_shuf=40):
    """raw tau, this member's own label-shuffle null, and the difference."""
    pred = -first[:, p, :]
    raw = float(kendall_tau_b(pred, Y).mean())
    nl = float(np.mean([
        kendall_tau_b(pred, Y[np.random.default_rng(k).permutation(Y.shape[0])]).mean()
        for k in range(n_shuf)]))
    return raw, nl, raw - nl


def order_stats(first, p):
    """Degeneracy of one member's predicted orders: distinct ticks, ties, silence, modal share."""
    f = first[:, p, :].astype(int)
    iu, ju = np.triu_indices(N_OUT, 1)
    ties = float((np.sign(f[:, iu] - f[:, ju]) == 0).mean())
    ranks = np.argsort(np.argsort(-f, axis=1), axis=1)       # earlier spike = higher rank
    _, cnt = np.unique(ranks, axis=0, return_counts=True)
    return dict(distinct_ticks=float(np.mean([len(np.unique(r)) for r in f])),
                tie_rate=ties,
                silent_frac=float((f >= N_TICKS).mean()),
                modal_order_share=float(cnt.max() / f.shape[0]),
                mean_first_tick=float(f[f < N_TICKS].mean()) if (f < N_TICKS).any() else -1.0,
                frac_in_input_phase=float((f < T_IN).mean()),
                frac_in_compute_phase=float(((f >= T_IN) & (f < 2 * T_IN)).mean()),
                frac_in_readout_phase=float(((f >= 2 * T_IN) & (f < N_TICKS)).mean()))


def weight_health(h, genomes):
    """Where the excitatory weights sit after training, straight off the device.

    The readout going silent could mean STDP wrote a selective association OR that the whole
    reservoir depressed itself to zero. These two look identical from the readout alone, so
    read the weights and settle it.
    """
    gs = [{k: v.copy() for k, v in g.items()} for g in genomes]
    ss.readback(h, gs)
    w = np.concatenate([g["weight"][g["src_pool"] != ss.INH] for g in gs])
    return dict(exc_w_mean=float(w.mean()), exc_w_median=float(np.median(w)),
                exc_w_max=float(w.max()), frac_below_0p1=float((w < 0.1).mean()),
                frac_below_1=float((w < 1.0).mean()))


def train(h, Xpool, Ypool, enc, batch, seed, n_batches, current, tcur, shuffle=False, tag="",
          trace=None):
    """n_batches of teacher-clamped STDP over the whole pool. shuffle destroys the pairing.

    `trace` = (every, fn) calls fn(i) after batch i whenever i is a checkpoint, so the run can
    watch activity collapse or survive instead of only seeing the end state.
    """
    t0 = time.time()
    for i in range(n_batches):
        Xb, Yb, _ = sample_batch(Xpool, Ypool, batch, seed, 900_000 + i)
        if shuffle:
            Yb = Yb[np.random.default_rng(7000 + i).permutation(Yb.shape[0])]
        run_episode(h, Xb, enc, current, train=True,
                    teacher_ticks=ss.teacher_ticks_for(Yb), teacher_current=tcur)
        if trace and (i + 1) in trace[0]:
            trace[1](i + 1)
    print(f"    [{tag}] {n_batches} clamped STDP batches in {time.time() - t0:.1f}s", flush=True)


def evaluate(h, Xv, Yv, enc, current, tcur, label, teacher=False, offset=64):
    """Score one net on held-out states, on BOTH readouts the design could mean.

    PLAIN is the specified metric: first spike anywhere in [0, 96). WINDOW is the same raster
    read only over [offset, 96), i.e. the phase the teacher clamp actually occupies. They are
    the same number only if the readout cells are silent until the readout phase -- which,
    being ordinary reservoir neurons, they are not. Reporting both is what tells us whether
    the specified metric is reading the association or the input volley washing through.
    """
    tt = ss.teacher_ticks_for(Yv) if teacher else None
    first, R = run_episode(h, Xv, enc, current, teacher_ticks=tt, teacher_current=tcur)
    win = R[..., offset:]
    fw = np.where(win.any(-1), win.argmax(-1) + offset, N_TICKS).astype(np.float64)
    rows = []
    for p in range(h["P"]):
        raw, nl, cor = corrected_tau(first, Yv, p)
        _, _, corw = corrected_tau(fw, Yv, p)
        rows.append(dict(member=p, raw_tau=raw, null=nl, corrected=cor, corrected_window=corw,
                         **order_stats(first, p)))
    cor = np.array([r["corrected"] for r in rows])
    corw = np.array([r["corrected_window"] for r in rows])
    rate = float(R.sum(-1).mean())                      # spikes per readout neuron per episode
    ph = [float(np.mean([r[k] for r in rows])) for k in
          ("frac_in_input_phase", "frac_in_compute_phase", "frac_in_readout_phase")]
    print(f"  {label:22s} tau[0,96) {cor.mean():+.4f}+/-{cor.std():.4f}   "
          f"tau[{offset},96) {corw.mean():+.4f}+/-{corw.std():.4f}   "
          f"spikes/cell {rate:.1f}   distinct ticks "
          f"{np.mean([r['distinct_ticks'] for r in rows]):.2f}   "
          f"1st-spike phase in/comp/out {ph[0]:.2f}/{ph[1]:.2f}/{ph[2]:.2f}   "
          f"modal order {np.mean([r['modal_order_share'] for r in rows]):.3f}", flush=True)
    return dict(label=label, members=rows, corrected_mean=float(cor.mean()),
                corrected_sd=float(cor.std()), corrected_best=float(cor.max()),
                corrected_window_mean=float(corw.mean()),
                corrected_window_best=float(corw.max()),
                spikes_per_cell=rate, phase_fracs=ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--train-batches", type=int, default=64)
    ap.add_argument("--stdp-lr", type=float, default=0.01)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--teacher-current", type=float, default=None)
    ap.add_argument("--teacher-offset", type=int, default=64)
    ap.add_argument("--teacher-levels", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fanout-scale", type=float, default=1.0,
                    help="divide every seed fan-out by this; it is the reservoir GAIN knob, "
                         "and therefore what sets the readout cells' firing rate. 1.0 (the "
                         "chapter default) gives ~10 spikes/cell/episode, which is not a "
                         "first-spike code at all")
    ap.add_argument("--out", default=None, help="write the full report as JSON here")
    a = ap.parse_args()

    tcur = a.current if a.teacher_current is None else a.teacher_current
    ss.ASSOC = True
    ss.ASSOC_TEACHER_OFFSET = a.teacher_offset
    ss.ASSOC_TEACHER_CURRENT = tcur
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y, Xpool, Ypool, Xval, Yval = load(a.batch, a.seed, a.n_val)
    enc = LatencyEncoder(Xpool)
    ss.fit_target_stats(Ypool, 2.5, a.teacher_levels)
    tgt = ss.target_offsets(Ypool)
    print(f"assoc sanity: K={a.pool}, teacher offset {a.teacher_offset} "
          f"(ticks {a.teacher_offset}..{a.teacher_offset + a.teacher_levels - 1}), "
          f"current {tcur}, stdp_lr {a.stdp_lr}, {a.train_batches} train batches, "
          f"{a.n_val} held-out states, dev {dev}")
    print(f"  pool target offsets: mean {tgt.mean():.2f} sd {tgt.std():.2f} "
          f"{len(np.unique(tgt))} levels; readout = exc {N_EXC - N_OUT}..{N_EXC - 1}")

    rep = dict(config=vars(a), teacher_current=tcur)

    fs = a.fanout_scale
    fan = dict(fanout_e=max(1, round(80 / fs)), fanout_i=max(1, round(20 / fs)),
               fanout_inh=max(1, round(100 / fs)), fanout_in=max(1, round(100 / fs)),
               fanin_out=max(1, round(100 / fs)))
    genomes = [ss.seed_genome(np.random.default_rng(a.seed * 100 + i), a.w_max, **fan)
               for i in range(a.pool)]
    n_to_readout = int(np.mean([
        ((g["tgt_pool"] == ss.EXC) & (g["tgt_idx"] >= N_EXC - N_OUT)).sum() for g in genomes]))
    n_in_to_readout = int(np.mean([
        ((g["tgt_pool"] == ss.EXC) & (g["tgt_idx"] >= N_EXC - N_OUT)
         & (g["src_pool"] == ss.INP)).sum() for g in genomes]))
    n_from_readout = int(np.mean([
        ((g["src_pool"] == ss.EXC) & (g["src_idx"] >= N_EXC - N_OUT)).sum() for g in genomes]))
    assert not (np.concatenate([g["tgt_pool"] for g in genomes]) == ss.OUTP).any()
    print(f"  genome: {genomes[0]['weight'].size:,} synapses, NO output-pool targets. "
          f"Per net the 6 readout cells receive {n_to_readout} synapses "
          f"({n_in_to_readout} straight from the input layer) and send {n_from_readout} "
          f"back into the reservoir.")
    rep["wiring"] = dict(n_synapses=int(genomes[0]["weight"].size),
                         readout_fan_in=n_to_readout,
                         readout_fan_in_from_input=n_in_to_readout,
                         readout_fan_out=n_from_readout)

    h = ss.build_pool(genomes, dev, seed=1, stdp_lr=a.stdp_lr, w_max=a.w_max)
    print(f"  built: {h['n_syn']:,} synapses, readout ids "
          f"{h['readout_ids'][:6].tolist()} ... ({h['readout_ids'].size} total)")

    # ---------------------------------------------------------------- (1) does the clamp bite
    print("\n(1) CLAMP -- teacher on, no STDP: do the readout cells fire at their encoded ticks?")
    Xb, Yb, _ = sample_batch(Xpool, Ypool, a.batch, a.seed, 0)
    tt = ss.teacher_ticks_for(Yb)
    _, R = run_episode(h, Xb, enc, a.current, train=False, teacher_ticks=tt,
                       teacher_current=tcur)
    win = R[..., a.teacher_offset:]                       # [B, P, 6, N_TICKS - offset]
    has = win.any(-1)
    fw = np.where(has, win.argmax(-1) + a.teacher_offset, -1)
    err = fw - tt[:, None, :]
    ok = has & (err == 0)
    within1 = has & (np.abs(err) <= 1)
    # Does the cell fire ON the teacher tick at all (not necessarily first), and is it already
    # spiking before the clamp arrives? The second number is the one that decides whether TTFS
    # can see the teacher: a cell that fired at tick 70 on its own cannot be "made first" at 80.
    on_tick = R[np.arange(R.shape[0])[:, None, None], np.arange(R.shape[1])[None, :, None],
                np.arange(N_OUT)[None, None, :], tt[:, None, :]] > 0
    before = np.array([[[R[b, p, j, :tt[b, j]].any() for j in range(N_OUT)]
                        for p in range(R.shape[1])] for b in range(R.shape[0])])
    clamp = dict(fired_in_window=float(has.mean()), exact=float(ok.mean()),
                 within_1_tick=float(within1.mean()),
                 fired_on_teacher_tick=float(on_tick.mean()),
                 already_spiking_before_teacher_tick=float(before.mean()),
                 spikes_per_cell=float(R.sum(-1).mean()),
                 mean_abs_err=float(np.abs(err[has]).mean()) if has.any() else -1.0,
                 median_err=float(np.median(err[has])) if has.any() else -1.0)
    print(f"  fires ON its teacher tick: {clamp['fired_on_teacher_tick']:.4f}   "
          f"but ALREADY spiking before it: "
          f"{clamp['already_spiking_before_teacher_tick']:.4f}   "
          f"spikes/cell/episode {clamp['spikes_per_cell']:.1f}")
    print(f"  first spike in [{a.teacher_offset}, {N_TICKS}) == teacher tick: "
          f"{clamp['exact']:.4f}   within 1: {clamp['within_1_tick']:.4f}   "
          f"mean |err| {clamp['mean_abs_err']:.2f}   median err {clamp['median_err']:+.1f}")
    rep["clamp"] = clamp

    # ---------------------------------------------------------------- (2)+(3) cold, then trained
    print("\n(2)+(3) EVAL -- teacher off, held-out states")
    def ev(*args, **kw):
        return evaluate(*args, offset=a.teacher_offset, **kw)

    rep["cold"] = ev(h, Xval, Yval, enc, a.current, tcur, "COLD (no STDP yet)")
    rep["teacher_on_control"] = ev(h, Xval, Yval, enc, a.current, tcur,
                                   "TEACHER-ON control", teacher=True)

    marks = sorted({m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
                    if m < a.train_batches})
    rep["trace"] = []

    def tick(i):
        r = ev(h, Xval, Yval, enc, a.current, tcur, f"  after {i} batches")
        r["batches"] = i
        rep["trace"].append(r)

    train(h, Xpool, Ypool, enc, a.batch, a.seed, a.train_batches, a.current, tcur, tag="paired",
          trace=(set(marks), tick))
    rep["trained"] = ev(h, Xval, Yval, enc, a.current, tcur, "TRAINED (paired)")
    rep["weights_after_training"] = weight_health(h, genomes)
    print(f"  weights after training: {rep['weights_after_training']}")

    # ---------------------------------------------------------------- (4) the trivial-solution control
    print("\n(4) TRIVIAL-SOLUTION control -- retrain a FRESH copy on SHUFFLED teacher ticks")
    del h
    torch.cuda.empty_cache()
    genomes2 = [ss.seed_genome(np.random.default_rng(a.seed * 100 + i), a.w_max, **fan)
                for i in range(a.pool)]
    h2 = ss.build_pool(genomes2, dev, seed=1, stdp_lr=a.stdp_lr, w_max=a.w_max)
    train(h2, Xpool, Ypool, enc, a.batch, a.seed, a.train_batches, a.current, tcur,
          shuffle=True, tag="shuffled")
    rep["trained_shuffled"] = ev(h2, Xval, Yval, enc, a.current, tcur, "TRAINED (shuffled)")

    # ---------------------------------------------------------------- verdict
    cold, tr, sh = (rep["cold"]["corrected_mean"], rep["trained"]["corrected_mean"],
                    rep["trained_shuffled"]["corrected_mean"])
    print(f"\nSUMMARY  cold {cold:+.4f}   trained {tr:+.4f}   shuffled-teacher {sh:+.4f}")
    print(f"         learning gain (trained - cold)      {tr - cold:+.4f}")
    print(f"         association gain (trained - shuffled) {tr - sh:+.4f}")
    rep["summary"] = dict(cold=cold, trained=tr, shuffled=sh,
                          learning_gain=tr - cold, association_gain=tr - sh,
                          teacher_on=rep["teacher_on_control"]["corrected_mean"])
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
