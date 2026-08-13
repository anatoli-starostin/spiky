"""Arm-A / arm-B ES sweep: (mu+lambda) elitist, batch resampled per generation, held-out val.

Protocol: pop P, elite kept, P-1 mutated children per generation. Training batch is redrawn
every generation from the training pool and shared by the whole population (so candidates
stay perfectly paired within a generation). The elite is then scored on a FIXED held-out
set. The null is computed once on a FIXED reference batch and used as the yardstick
throughout — headline metric is val (tau - null).

    python es_sweep.py --arm a --pop 16 --gens 40 --seeds 0 1 2
    python es_sweep.py --arm b --npz results/reservoir_b_3600s.npz ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from es_harness import (LatencyEncoder, N_EXC, N_TICKS, build, fitness, kendall_tau_b,
                        mutate, random_genome, reservoir_wiring, run_episode)
from es_smoke import load, sample_batch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

# Live Slack progress bar. A file rendezvous under ~/.cache/slack_facade/progress — no
# network from this side, so it stays cage-safe and costs no approval.
sys.path.insert(0, "/home/astarostin/work/slack-facade")
try:
    import progress as _prog
except Exception:
    _prog = None


def liveness(h, X, enc, current):
    from spiky.spnet.spnet import NeuronDataType
    first, _ = run_episode(h, X[:64], enc, current)
    R = h["spnet"].export_neuron_data(
        torch.tensor(h["ids"][0], dtype=torch.int32, device=h["device"]),
        64, NeuronDataType.Spike, 0, N_TICKS - 1).cpu().numpy()
    of = first < N_TICKS
    return dict(spikes=float(R.sum() / 64),
                recruited=float(R.any(-1).sum() / 64 / N_EXC * 100),
                out_fired=float(of.mean() * 100))


def own_null(first, Y, n_shuf=200):
    """Chance level for THESE predictions: shuffle Y against this model's own orderings.

    The previous metric took one null from a gen-0 random-wiring net and subtracted it from
    every evolved model. But the label-shuffle null is a property of the MODEL's predicted
    orderings (a near-constant ordering scores well against the average target ordering
    regardless of input), so it must be recomputed for whatever predictions are being
    scored. Measured cost of getting this wrong: per-seed nulls of +0.18/-0.10/-0.15 on
    networks whose raw tau agreed to +-0.05.
    """
    pred = -first[:, 0, :]
    return float(np.mean([kendall_tau_b(pred, Y[np.random.default_rng(k).permutation(Y.shape[0])]).mean()
                          for k in range(n_shuf)]))


def score_val(h, Xval, Yval, enc, current):
    """-> (raw tau, own null, corrected) for one model on the held-out set."""
    first, _ = run_episode(h, Xval, enc, current)
    tau = float(kendall_tau_b(-first[:, 0, :], Yval).mean())
    nl = own_null(first, Yval)
    return tau, nl, tau - nl


def run_seed(seed, a, dev, edges, bar=None, si=0, t_start=None):
    X, Y, Xpool, Ypool, Xval, Yval = load(a.batch, seed, a.n_val)
    enc = LatencyEncoder(Xpool)
    res = reservoir_wiring(np.random.default_rng(1234 + seed))     # reservoir seed varies
    rng = np.random.default_rng(seed)
    kw = dict(res_edges=edges) if edges is not None else dict(res_scale=1.0)
    # weights-only: wiring (in_tgt/out_src) and delays (in_dly/out_dly) are drawn once at
    # init and frozen thereafter, exactly like the reservoir. Only in_w/out_w mutate.
    mkw = dict(p_rewire=0.0, p_delay=0.0) if a.weights_only else {}

    h0 = build([random_genome(np.random.default_rng(seed), a.w_max)], res, dev, **kw)
    live0 = liveness(h0, X, enc, a.current)
    del h0

    elite = random_genome(rng, a.w_max)
    pop = [elite] + [random_genome(rng, a.w_max) for _ in range(a.pop - 1)]
    hist, t0 = [], time.time()
    for gen in range(0, a.gens + 1):
        Xb, Yb, _ = sample_batch(Xpool, Ypool, a.batch, seed, gen)
        if gen > 0:
            pop = [elite] + [mutate(elite, rng, a.w_max, **mkw) for _ in range(a.pop - 1)]
        hk = build(pop, res, dev, **kw)
        fk, _, _ = fitness(hk, Xb, Yb, enc, a.current)
        elite = pop[int(fk.argmax())]
        del hk
        hv = build([elite], res, dev, **kw)
        v_raw, v_null, v_corr = score_val(hv, Xval, Yval, enc, a.current)
        del hv
        hist.append(dict(gen=gen, train=float(fk.max()), train_mean=float(fk.mean()),
                         val=v_raw, val_null=v_null, val_corr=v_corr))
        if gen % 5 == 0 or gen == a.gens:
            print(f"    gen {gen:3d}  train {fk.max():+.4f}  mean {fk.mean():+.4f}  "
                  f"VAL raw {v_raw:+.4f}  own-null {v_null:+.4f}  CORR {v_corr:+.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            if bar and _prog:
                # GLOBAL step across the whole multi-seed run, so the bar advances once
                # from 0 to 100% rather than resetting per seed.
                step = si * (a.gens + 1) + gen + 1
                total = len(a.seeds) * (a.gens + 1)
                el = time.time() - t_start
                eta = el / max(step, 1) * (total - step)
                best_so_far = max(h["val_corr"] for h in hist)
                _prog.progress_update(
                    bar, step=step, total=total,
                    stats=f"seed {si+1}/{len(a.seeds)} · gen {gen}/{a.gens} · "
                          f"best CORR {best_so_far:+.4f} · eta ~{eta/60:.1f}m")
    hL = build([elite], res, dev, **kw)
    liveN = liveness(hL, X, enc, a.current)
    del hL
    return dict(seed=seed, hist=hist, live_first=live0, live_last=liveN,
                best_val=max(h["val"] for h in hist),
                best_corr=max(h["val_corr"] for h in hist),
                final_val=hist[-1]["val"], final_corr=hist[-1]["val_corr"],
                gen0_val=hist[0]["val"], gen0_corr=hist[0]["val_corr"],
                wall_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="a", choices=["a", "b"])
    ap.add_argument("--npz", default=os.path.join(OUT, "reservoir_b_3600s.npz"))
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--task", default=None, help="BODY_TASK id: posts a live progress bar in that thread")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    edges = None
    if a.arm == "b":
        Z = np.load(a.npz, allow_pickle=True)
        edges = (Z["src_local"], Z["src_is_inh"], Z["tgt_local"], Z["tgt_is_inh"],
                 Z["weight"], Z["delay"])
        print(f"arm B: {edges[0].shape[0]:,} synapses from {os.path.basename(a.npz)}")
    print(f"ARM {a.arm.upper()}  pop {a.pop}  gens {a.gens}  batch {a.batch}  "
          f"val {a.n_val}  seeds {a.seeds}  device {dev}")

    bar = None
    if a.task and _prog:
        bar = _prog.progress_start(
            f"arm {a.arm.upper()} ES · pop {a.pop} · {a.gens} gens · {len(a.seeds)} seeds",
            task=a.task)
        print(f"progress bar handle {bar} -> task {a.task}", flush=True)

    runs, t_start = [], time.time()
    try:
        for si, s in enumerate(a.seeds):
            print(f"\n--- seed {s} ---", flush=True)
            r = run_seed(s, a, dev, edges, bar=bar, si=si, t_start=t_start)
            runs.append(r)
            print(f"    best val raw {r['best_val']:+.4f}  best CORR {r['best_corr']:+.4f}"
                  f"  {r['wall_s']:.0f}s", flush=True)
            json.dump(runs, open(os.path.join(
                OUT, f"es_sweep_arm{a.arm}{a.tag}.json"), "w"), indent=1)
    except BaseException:
        # ALWAYS terminate the bar, or a dead poller leaves it stale forever.
        if bar and _prog:
            _prog.progress_done(bar, ok=False, final_text="run failed — see log")
        raise

    def agg(key):
        v = np.array([r[key] for r in runs])
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), np.round(v, 4)

    print(f"\n=== ARM {a.arm.upper()} SUMMARY over {len(runs)} seeds ===")
    for label, key in (("gen-0  raw", "gen0_val"), ("final  raw", "final_val"),
                       ("best   raw", "best_val"), ("gen-0  CORR", "gen0_corr"),
                       ("final  CORR", "final_corr"), ("best   CORR", "best_corr")):
        m, sd, v = agg(key)
        print(f"  {label:12s}: {m:+.4f} +- {sd:.4f}   {v}")
    for r in runs:
        b = r["best_corr"]
        gen_at = next(h["gen"] for h in r["hist"] if h["val_corr"] >= 0.95 * b)
        print(f"  seed {r['seed']}: reaches 95% of its best by gen {gen_at}; "
              f"liveness first {r['live_first']['out_fired']:.0f}% out / "
              f"{r['live_first']['recruited']:.0f}% exc, last "
              f"{r['live_last']['out_fired']:.0f}% out / {r['live_last']['recruited']:.0f}% exc")

    if bar and _prog:
        m, sd, _ = agg("best_corr")
        _prog.progress_done(bar, ok=True,
                            final_text=f"best corrected val {m:+.4f} +- {sd:.4f} "
                                       f"over {len(runs)} seeds")


if __name__ == "__main__":
    main()
