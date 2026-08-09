"""Score a saved checkpoint's best member on the held-out set, in a fresh process.

WHY THIS EXISTS. `steady_state.py` evaluates the best genome on the held-out set as its very
last act, inside the same process that just ran the training pool. At large meta counts that
allocation can fail even though the training itself fit: exp004 (96 metas, K=32) completed all
300 rounds and then died with `cudaErrorMemoryAllocation` in `build_pool -> add_connections`,
and 40 supervisor restarts died identically, because each resumed run rebuilt the training pool
before reaching the evaluation again.

**IT WAS NOT ACTUALLY OUT OF MEMORY.** Running this standalone with 29 GB of the GPU free
reproduced the identical failure, which is what exposed the real cause. `steady_state.py`'s
final evaluation line is

    hb = build_pool([genomes[best_i]], dev, seed=1)          # steady_state.py:816

and it passes neither `stdp_lr` nor `w_max`. `build_pool` selects its meta bank on that
argument: `stdp_lr > 0` builds `stage2_metas` — 2 x (D_MAX - D_MIN + 1) metas, plastic
excitatory bank plus a frozen inhibitory bank, with inhibitory synapses offset by
`N_DELAY_METAS` — while `stdp_lr == 0` builds `harness.delay_metas()`, whose `d_max` default
was bound at import from `harness.D_MAX = 20` and therefore ignores `--d-max` entirely, and
which applies no inhibitory offset.

For exp004 (d_max 48) that means the genome carries meta indices up to 95 into a network with
only 20 metas. The engine dereferences out of range and asks the allocator for a nonsense size,
which surfaces as `cudaErrorMemoryAllocation` — an addressing bug wearing an OOM's clothes. It
also explains why 40 supervisor restarts failed identically on a machine with plenty of memory:
memory was never the problem.

So this script must be told the run's delay range and whether it used STDP, and it rebinds
`D_MAX`/`N_DELAY_METAS` exactly as `steady_state.main()` does before building anything.

    python eval_heldout.py --ckpt .../ck_delay148.npz --d-max 48 --stdp-lr 0.01

Everything else reproduces the in-run evaluation exactly: same `load(batch, seed, n_val)`
split, same `LatencyEncoder(Xpool)`, `build_pool([best], seed=1)`, same `score(...)`. Defaults
match steady_state.py's, so a run that used a non-default `--current` / `--tie-penalty` /
`--w-max` must pass them here too.
"""
import argparse
import json
import os

import numpy as np
import torch

import steady_state as S
from data import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ck_<tag>.npz")
    ap.add_argument("--member", type=int, default=None,
                    help="pool index to score (default: highest EWMA, as the in-run eval does)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--tie-penalty", type=float, default=0.0)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--d-max", type=int, default=S.D_MAX,
                    help="MUST match the run: it sizes the meta bank the genome indexes into")
    ap.add_argument("--stdp-lr", type=float, default=0.0,
                    help="MUST match the run (>0 vs 0). Selects stage2_metas (plastic exc bank "
                         "+ frozen inh bank, inh offset by N_DELAY_METAS) vs harness."
                         "delay_metas (single bank, no offset). The value itself is inert here "
                         "-- the held-out episode runs with train=False.")
    ap.add_argument("--n-exc", type=int, default=S.N_EXC,
                    help="MUST match the run: build_pool sizes the neuron pools from this and "
                         "the genome's indices are relative to it")
    ap.add_argument("--n-inh", type=int, default=S.N_INH)
    ap.add_argument("--use-drive", action="store_true",
                    help="apply the member's evolved output-drive scale from the checkpoint. "
                         "The in-run evaluation did NOT (it calls build_pool with drives=None), "
                         "so leave this off to reproduce a recorded number and turn it on to "
                         "score the net that was actually selected.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="evaluate this many times and report mean +/- std. USE THIS. Building "
                         "the same genome twice does not give the same network: measured spread "
                         "on corrected tau is ~0.03-0.05, which is larger than most differences "
                         "between experiments. A single draw is not a measurement.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    a = ap.parse_args()

    # Same rebinding steady_state.main() does. build_pool and stage2_metas read these globals,
    # so they have to be set before the genome's meta indices mean anything.
    S.D_MAX = a.d_max
    S.N_DELAY_METAS = a.d_max - S.D_MIN + 1
    S.N_EXC, S.N_INH = a.n_exc, a.n_inh
    S.POOL_SIZE = {S.EXC: a.n_exc, S.INH: a.n_inh, S.INP: S.N_IN, S.OUTP: S.N_OUT}

    genomes, ewma, age, newborn, hist, next_rnd, _rng, lineage, drives = S.load_ckpt(a.ckpt)
    best_i = a.member if a.member is not None else int(np.nanargmax(ewma))
    n_syn = int(genomes[best_i]["weight"].size)
    print(f"checkpoint {a.ckpt}\n  {len(genomes)} genomes, next_rnd {next_rnd}, "
          f"{len(hist)} rounds of history", flush=True)
    print(f"  scoring member {best_i}: EWMA {ewma[best_i]:+.4f}, {n_syn:,} synapses, "
          f"delays [{int(genomes[best_i]['delay'].min())}, "
          f"{int(genomes[best_i]['delay'].max())}]", flush=True)

    _X, _Y, Xpool, _Yp, Xval, Yval = load(a.batch, a.seed, a.n_val)
    enc = S.LatencyEncoder(Xpool)
    print(f"  held-out set: {Xval.shape[0]} states (the last 4000 samples are never "
          f"trained on)", flush=True)

    n_metas = 2 * S.N_DELAY_METAS if a.stdp_lr > 0 else S.N_DELAY_METAS
    print(f"  building {n_metas} metas (d_max {a.d_max}, "
          f"{'stage2 plastic+frozen banks' if a.stdp_lr > 0 else 'single frozen bank'})",
          flush=True)
    dr = np.array([float(drives[best_i])]) if a.use_drive else None
    if dr is not None:
        print(f"  applying evolved drive {dr[0]:.4f}", flush=True)

    draws = []
    for rep in range(a.repeats):
        hb = S.build_pool([genomes[best_i]], a.device, seed=1, stdp_lr=a.stdp_lr,
                          w_max=a.w_max, drives=dr)
        fit, first, ties = S.score(hb, Xval, Yval, enc, a.current, a.tie_penalty)
        pred = -first[:, 0, :]
        raw = float(S.kendall_tau_b(pred, Yval).mean())
        t = first[:, 0, :]
        draws.append(dict(
            corrected_tau=float(fit[0]), raw_tau=raw, own_null=raw - float(fit[0]),
            tie_rate=float(ties[0]),
            distinct_ticks_per_state=float(np.mean([len(np.unique(r)) for r in t])),
            tick_mean=float(t.mean()), tick_std=float(t.std()),
            tick_min=int(t.min()), tick_max=int(t.max()),
            silent_frac=float((t >= S.N_TICKS).mean())))
        print(f"  draw {rep + 1}/{a.repeats}: corrected {draws[-1]['corrected_tau']:+.4f}  "
              f"raw {raw:+.4f}  tie {draws[-1]['tie_rate']:.4f}  "
              f"ticks/state {draws[-1]['distinct_ticks_per_state']:.3f}", flush=True)
        del hb
        torch.cuda.empty_cache()

    def agg(field):
        v = np.array([d[field] for d in draws], float)
        return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    min=float(v.min()), max=float(v.max()))

    res = dict(
        checkpoint=os.path.basename(a.ckpt), member=best_i, ewma=float(ewma[best_i]),
        n_synapses=n_syn, n_val=int(Xval.shape[0]),
        d_max=a.d_max, stdp_lr=a.stdp_lr, n_metas=n_metas,
        drive_applied=float(dr[0]) if dr is not None else None,
        repeats=a.repeats, draws=draws,
        heldout_corrected_tau=agg("corrected_tau"), heldout_raw_tau=agg("raw_tau"),
        heldout_own_null=agg("own_null"), heldout_tie_rate=agg("tie_rate"),
        distinct_first_spike_ticks_per_state=agg("distinct_ticks_per_state"))

    c = res["heldout_corrected_tau"]
    print(f"\nMEMBER {best_i}: EWMA {ewma[best_i]:+.4f}  ({n_syn:,} synapses)")
    print(f"  HELD-OUT corrected tau  {c['mean']:+.4f} +/- {c['std']:.4f}  "
          f"over {a.repeats} build(s), range [{c['min']:+.4f}, {c['max']:+.4f}]")
    for f in ("heldout_raw_tau", "heldout_own_null", "heldout_tie_rate",
              "distinct_first_spike_ticks_per_state"):
        g = res[f]
        print(f"  {f:38s} {g['mean']:+.4f} +/- {g['std']:.4f}")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
