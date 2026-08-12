"""exp012 pre-flight: is the tiny direct-genome substrate valid, and can fitness move?

Six checks, each one a thing that could silently be wrong:

  1 GENOME     random_genome and 200 rounds of mutate keep Dale's law and the two delay
               ranges exact -- asserted, because the law is claimed to be structural.
  2 BUILD      P candidates pack into one net; every synapse's weight (INCLUDING the negative
               ones) and delay come back out of the compiled net unchanged.
  3 ISOLATION  a candidate scored packed alongside 31 others scores identically to the same
               candidate scored alone. If the id blocks leaked this would be silently false
               and every number in the experiment would be meaningless.
  4 ALIVE      at round 0, over a sweep of w_max: do outputs fire in the readout window at
               all, and do they take more than one distinct value? A pool that is silent or
               constant is degenerate and selection has nothing to work on.
  5 SPREAD     is there real variance in fitness across a random pool? Selection needs
               something to select.
  6 MOVE       50 rounds of a throwaway hill-climb -- does MSE actually go down?

Run:  python tiny_preflight.py [--w-max 40] [--out sanity/preflight.json]
"""
import argparse
import json
import os
import time

import numpy as np

import tiny_snn as T
from data import load
from harness import LatencyEncoder


def check_genome(rng, w_max, n_rounds=200):
    g = T.random_genome(rng, w_max)
    tot = np.array([T.dale_ok(g)[1], T.delays_ok(g)[1], T.legal_ok(g)[1], T.pins_ok(g)[1]])
    struct = [T.structure_counts(g)]
    for _ in range(n_rounds):
        g = T.mutate(g, rng, w_max)
        tot += [T.dale_ok(g)[1], T.delays_ok(g)[1], T.legal_ok(g)[1], T.pins_ok(g)[1]]
        struct.append(T.structure_counts(g))
    d = g["delay"][g["mask"]]
    hid = g["mask"].copy()
    hid[:, T.COL_OUT] = False
    out = g["mask"].copy()
    out[:, T.COL_EXC] = False
    out[:, T.COL_INH] = False
    # the invariants, summed over every genome the chain produced -- not just the last one
    return dict(dale_violations=int(tot[0]), delay_violations=int(tot[1]),
                illegal_edge_violations=int(tot[2]), pinned_delay_violations=int(tot[3]),
                n_legal_cells=int(T.LEGAL.sum()), n_cells=int(T.LEGAL.size),
                configured_range=[T.D_LO, T.D_HI], n_metas=int(T.N_METAS),
                max_in_to_inh_over_chain=max(s["n_in_to_inh"] for s in struct),
                max_inh_to_out_over_chain=max(s["n_inh_to_out"] for s in struct),
                max_in_to_out_over_chain=max(s["n_in_to_out"] for s in struct),
                inh_to_exc_delays_seen=sorted({d for s in struct
                                               for d in s["inh_to_exc_delays"]}),
                structure_after_200=struct[-1],
                hidden_delay_range=[int(g["delay"][hid].min()), int(g["delay"][hid].max())],
                output_delay_range=[int(g["delay"][out].min()), int(g["delay"][out].max())],
                delay_span_seen=[int(d.min()), int(d.max())],
                after_200_mutations=T.genome_stats(g),
                ok=bool(tot.sum() == 0
                        and max(s["n_in_to_inh"] for s in struct) == 0
                        and max(s["n_inh_to_out"] for s in struct) == 0
                        and max(s["n_in_to_out"] for s in struct) == 0))


def check_crossover(rng, w_max, n_pairs=200):
    """200 random parent pairs -> crossover -> chained mutation. Nothing may break.

    Two distinct things are asserted. (a) The invariants survive recombination: Dale, the
    delay range and edge legality all hold on the child AND after the child is mutated.
    (b) BUNDLE COHERENCE -- every child cell's (mask, weight, delay) came from ONE parent.
    (b) is the one that would fail silently: a child that mixed a parent's weight with the
    other parent's delay would still pass every invariant check and still run, it would just
    be inheriting a weight tuned against a delay it does not have.
    """
    tot = np.zeros(4)
    mix = pins = struct = 0
    frac = []
    for _ in range(n_pairs):
        g1 = T.random_genome(rng, w_max)
        g2 = T.random_genome(rng, w_max)
        before = ({k: v.copy() for k, v in g1.items()}, {k: v.copy() for k, v in g2.items()})
        c = T.crossover(g1, g2, rng)
        tot += [T.dale_ok(c)[1], T.delays_ok(c)[1], T.legal_ok(c)[1],
                T.bundle_coherent(c, g1, g2)[1]]
        pins += T.pins_ok(c)[1]
        sc = T.structure_counts(c)
        struct += sc["n_in_to_inh"] + sc["n_inh_to_out"] + sc["n_in_to_out"]
        # parents must be untouched -- crossover has copy semantics
        for a_, b_ in zip((g1, g2), before):
            for k in a_:
                mix += int(not np.array_equal(a_[k], b_[k]))
        m = T.mutate(c, rng, w_max)
        tot += [T.dale_ok(m)[1], T.delays_ok(m)[1], T.legal_ok(m)[1], 0]
        pins += T.pins_ok(m)[1]
        sm = T.structure_counts(m)
        struct += sm["n_in_to_inh"] + sm["n_inh_to_out"] + sm["n_in_to_out"]
        # Is the coin fair? Only over cells where the parents actually DIFFER -- cells they
        # agree on carry no provenance, and counting them would report ~0.82 regardless of
        # whether crossover works, which is exactly the kind of number that looks fine and
        # means nothing.
        d = ~(np.equal(g1["mask"], g2["mask"]) & np.equal(g1["delay"], g2["delay"])
              & np.equal(g1["weight"], g2["weight"]))
        s1 = (np.equal(c["mask"], g1["mask"]) & np.equal(c["delay"], g1["delay"])
              & np.equal(c["weight"], g1["weight"]))
        frac.append(float(s1[d].mean()) if d.any() else 0.5)
    return dict(n_pairs=n_pairs, dale_violations=int(tot[0]), delay_violations=int(tot[1]),
                illegal_edge_violations=int(tot[2]), incoherent_cells=int(tot[3]),
                parents_mutated=int(mix),
                pinned_delay_violations=int(pins),
                illegal_structure_cells=int(struct),
                from_parent1_over_differing_cells=float(np.mean(frac)),
                from_parent1_sd=float(np.std(frac)),
                ok=bool(tot.sum() == 0 and mix == 0 and pins == 0 and struct == 0
                        and abs(np.mean(frac) - 0.5) < 0.02))


def check_build(genomes, device, w_ceiling):
    H = T.build(genomes, device=device, w_ceiling=w_ceiling)
    rt = T.verify_round_trip(H)
    rt["ok"] = bool(rt["missing"] == 0
                    and rt["weights_ok"] == rt["n_requested"]
                    and rt["delays_ok"] == rt["n_requested"]
                    and rt["negative_ok"] == rt["n_negative"])
    return H, rt


def check_isolation(genomes, X, Y, enc, device, w_ceiling, which=(0, 7, 19, 31)):
    H = T.build(genomes, device=device, w_ceiling=w_ceiling)
    packed = T.score(H, X, Y, enc)["mse"]
    solo = []
    for i in which:
        Hs = T.build([genomes[i]], device=device, w_ceiling=w_ceiling)
        solo.append(float(T.score(Hs, X, Y, enc)["mse"][0]))
    got = [float(packed[i]) for i in which]
    return dict(which=list(which), packed=got, solo=solo,
                max_abs_diff=float(np.max(np.abs(np.array(got) - np.array(solo)))),
                ok=bool(np.allclose(got, solo, atol=1e-9)))


def check_alive(rng, X, Y, enc, device, w_ceiling, w_grid, P=32):
    rows = []
    for w_max in w_grid:
        gs = [T.random_genome(rng, w_max) for _ in range(P)]
        H = T.build(gs, device=device, w_ceiling=w_ceiling)
        s = T.score(H, X, Y, enc)
        rows.append(dict(w_max=float(w_max),
                         silent_frac=float(s["silent"].mean()),
                         n_distinct_mean=float(s["n_distinct"].mean()),
                         mse_mean=float(s["mse"].mean()),
                         mse_best=float(s["mse"].min()),
                         mse_std_across_pool=float(s["mse"].std()),
                         tau_best=float(s["tau"].max())))
    return rows


def hill_climb(rng, X, Y, enc, device, w_max, w_ceiling, rounds=50, P=32):
    """Throwaway (1+P) hill-climb -- the cheapest possible test that fitness can move."""
    parent = T.random_genome(rng, w_max)
    H = T.build([parent], device=device, w_ceiling=w_ceiling)
    best = float(T.score(H, X, Y, enc)["mse"][0])
    hist = [best]
    for _ in range(rounds):
        kids = [T.mutate(parent, rng, w_max) for _ in range(P)]
        H = T.build(kids, device=device, w_ceiling=w_ceiling)
        m = T.score(H, X, Y, enc)["mse"]
        i = int(np.argmin(m))
        if m[i] < best:
            best, parent = float(m[i]), kids[i]
        hist.append(best)
    return hist, parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-max", type=float, default=40.0)
    ap.add_argument("--w-ceiling", type=float, default=200.0)
    ap.add_argument("--w-grid", type=float, nargs="*",
                    default=[5, 10, 20, 30, 40, 60, 90, 130])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    Xr, Yr, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    X, Y = Xr, Yr
    base = T.constant_baseline(Y)
    base_v = T.constant_baseline(Yv)

    out = dict(config=vars(a), constant_baseline_batch=base, constant_baseline_val=base_v,
               exp009_reference=dict(constant=39.19, best_stdp_800exc=37.52))

    print("1 GENOME  ", flush=True)
    out["genome"] = check_genome(rng, a.w_max)
    print("   ", out["genome"], flush=True)

    print("1b CROSSOVER", flush=True)
    out["crossover"] = check_crossover(rng, a.w_max)
    print("   ", out["crossover"], flush=True)

    gs = [T.random_genome(rng, a.w_max) for _ in range(a.pool)]

    print("2 BUILD   ", flush=True)
    H, rt = check_build(gs, a.device, a.w_ceiling)
    out["build"] = rt
    print("   ", rt, flush=True)

    print("3 ISOLATION", flush=True)
    out["isolation"] = check_isolation(gs, X, Y, enc, a.device, a.w_ceiling)
    print("   ", out["isolation"], flush=True)

    print("4 ALIVE (w_max sweep)", flush=True)
    out["alive"] = check_alive(rng, X, Y, enc, a.device, a.w_ceiling, a.w_grid, P=a.pool)
    for r in out["alive"]:
        print(f"    w_max {r['w_max']:6.1f}  silent {r['silent_frac']:.3f}  "
              f"distinct {r['n_distinct_mean']:5.2f}  mse {r['mse_mean']:7.2f} "
              f"(best {r['mse_best']:7.2f}, sd {r['mse_std_across_pool']:6.2f})  "
              f"tau_best {r['tau_best']:+.4f}", flush=True)

    print("5 SPREAD  ", flush=True)
    s = T.score(H, X, Y, enc)
    out["spread"] = dict(mse_mean=float(s["mse"].mean()), mse_std=float(s["mse"].std()),
                         mse_min=float(s["mse"].min()), mse_max=float(s["mse"].max()),
                         silent_frac=float(s["silent"].mean()),
                         ok=bool(s["mse"].std() > 1e-6))
    print("   ", out["spread"], flush=True)

    print("6 MOVE (hill-climb)", flush=True)
    hist, best_g = hill_climb(rng, X, Y, enc, a.device, a.w_max, a.w_ceiling,
                              rounds=a.rounds, P=a.pool)
    Hb = T.build([best_g], device=a.device, w_ceiling=a.w_ceiling)
    sv = T.score(Hb, Xv, Yv, enc)
    out["move"] = dict(history=[float(h) for h in hist],
                       start=float(hist[0]), end=float(hist[-1]),
                       improvement=float(hist[0] - hist[-1]),
                       heldout_mse=float(sv["mse"][0]),
                       heldout_tau=float(sv["tau"][0]),
                       heldout_mse_action=float(sv["mse_action"][0]),
                       heldout_silent=float(sv["silent"][0]),
                       genome=T.genome_stats(best_g),
                       ok=bool(hist[-1] < hist[0] - 1e-9))
    print(f"    batch MSE {hist[0]:.3f} -> {hist[-1]:.3f}   "
          f"held-out {sv['mse'][0]:.3f} (constant {base_v:.2f})  "
          f"tau {sv['tau'][0]:+.4f}  silent {sv['silent'][0]:.3f}", flush=True)

    out["elapsed_s"] = time.time() - t0
    out["all_ok"] = bool(out["genome"]["ok"] and out["crossover"]["ok"]
                         and out["build"]["ok"]
                         and out["isolation"]["ok"] and out["spread"]["ok"]
                         and out["move"]["ok"])
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(T.jsonable(out), f, indent=1)
        print(f"wrote {a.out}", flush=True)
    print(f"ALL_OK={out['all_ok']}  ({out['elapsed_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
