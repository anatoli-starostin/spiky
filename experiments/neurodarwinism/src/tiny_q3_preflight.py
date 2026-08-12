"""exp012: validate the 3-value weight grid and the odd-delay grid together.

Weights: {-0.5, 0, 1.0}, Dale-split to exc {0, 1.0} and inh {-0.5, 0} -- so every synapse is
BINARY within its sign class, with asymmetric E/I magnitudes.
Delays: the odd set {1, 3, ..., 63}, 32 levels, spacing 2, with the inh->exc pin at 1 (which
is itself a legal level, so the pin and the grid do not fight).

  a WEIGHTS   every weight exactly on the 3-value grid, Dale-split correct
  b DELAYS    every delay odd and in [1,63]; the pinned inh->exc delay is exactly 1
  c INVARIANTS legality, lateral-inhib motif, fan-out cap 16, active mask
  d AFFINE    the calibration genes stay continuous and near identity at init
  e HOP       what the weight hop actually does on a 2-level sub-grid
  f DEFAULTS  with no grids configured, the 11-level / free-integer behaviour is unchanged
"""
import argparse
import json
from collections import Counter

import numpy as np

import tiny_grow as G
import tiny_snn as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--fanout-cap", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    W3 = [-0.5, 0.0, 1.0]
    ODD = list(range(1, 64, 2))
    G.set_weight_levels(W3)
    G.set_delay_levels(ODD)
    G.QUANTIZED = True
    G.FANOUT_CAP = a.fanout_cap
    R = dict(weight_levels=G.QUANT_LEVELS.tolist(),
             exc_subgrid=G.QUANT_POS.tolist(), inh_subgrid=G.QUANT_NEG.tolist(),
             delay_levels=[int(ODD[0]), int(ODD[1]), "...", int(ODD[-1])],
             n_delay_levels=len(ODD), fanout_cap=a.fanout_cap)

    # ---------------------------------------------------------------- a WEIGHTS
    off = dale = 0
    exc_seen, inh_seen = Counter(), Counter()
    npos = G.N_IN + G.N_EXC_MAX
    g = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    for _ in range(a.trials):
        g = G.mutate(g, rng, p_grow=0, p_shrink=0)
        off += G.on_grid(g)[1]
        dale += G.dale_ok(g)[1]
        for v in np.round(g["weight"][:npos][g["mask"][:npos]], 10):
            exc_seen[float(v)] += 1
        for v in np.round(g["weight"][npos:][g["mask"][npos:]], 10):
            inh_seen[float(v)] += 1
    R["weights"] = dict(off_grid=off, dale_violations=dale,
                        exc_values_seen=sorted(exc_seen), inh_values_seen=sorted(inh_seen),
                        exc_subgrid_ok=bool(set(exc_seen).issubset({0.0, 1.0})),
                        inh_subgrid_ok=bool(set(inh_seen).issubset({-0.5, 0.0})),
                        ok=bool(off == 0 and dale == 0
                                and set(exc_seen).issubset({0.0, 1.0})
                                and set(inh_seen).issubset({-0.5, 0.0})))
    print("a WEIGHTS", json.dumps(T.jsonable(R["weights"]), indent=1), flush=True)

    # ---------------------------------------------------------------- b DELAYS
    d = g["delay"][g["mask"]]
    pin_present = g["mask"] & G.PIN_DELAY
    R["delays"] = dict(min=int(d.min()), max=int(d.max()),
                       all_odd=bool(bool((d % 2 == 1).all())),
                       in_range=bool(d.min() >= 1 and d.max() <= 63),
                       off_grid=G.delays_on_grid(g)[1],
                       n_distinct=int(len(np.unique(d))),
                       pinned_synapses=int(pin_present.sum()),
                       pinned_all_equal_1=bool((g["delay"][pin_present] == 1).all()),
                       pins_violations=G.pins_ok(g)[1],
                       ok=bool((d % 2 == 1).all() and d.min() >= 1 and d.max() <= 63
                               and G.delays_on_grid(g)[1] == 0 and G.pins_ok(g)[1] == 0))
    print("b DELAYS", json.dumps(T.jsonable(R["delays"]), indent=1), flush=True)

    # ---------------------------------------------------------------- c INVARIANTS
    tot = {}
    n = 0
    worst_fan = 0

    def acc(x):
        nonlocal n, worst_fan
        for k, v in G.all_ok(x).items():
            tot[k] = tot.get(k, 0) + v
        worst_fan = max(worst_fan, int(G.hidden_fanout(x).max()))
        n += 1

    g2 = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    for _ in range(a.trials):
        g2 = G.mutate(g2, rng, p_grow=0, p_shrink=0)
        acc(g2)
    for _ in range(a.trials // 2):
        acc(G.crossover(G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10),
                        G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10), rng))
    g3 = G.random_genome(rng, n_exc=4, n_inh=1)
    for _ in range(40):
        G.grow(g3, rng)
        acc(g3)
    R["invariants"] = dict(genomes_checked=n, violations=tot, worst_hidden_fanout=worst_fan,
                           ok=bool(sum(tot.values()) == 0 and worst_fan <= a.fanout_cap))
    print("c INVARIANTS", json.dumps(T.jsonable(R["invariants"]), indent=1), flush=True)

    # ---------------------------------------------------------------- d AFFINE
    g4 = G.random_genome(rng, n_exc=8, n_inh=2)
    a0, b0 = G.affine_of(g4)
    for _ in range(100):
        g4 = G.mutate(g4, rng)
    aa, bb = G.affine_of(g4)
    R["affine"] = dict(identity_at_init=bool(np.all(a0 == 1.0) and np.all(b0 == 0.0)),
                       a_after=np.round(aa, 4).tolist(), b_after=np.round(bb, 4).tolist(),
                       still_continuous=bool(not np.all(np.isin(np.round(aa, 10),
                                                                G.QUANT_LEVELS))),
                       ok=True)
    R["affine"]["ok"] = bool(R["affine"]["identity_at_init"]
                             and R["affine"]["still_continuous"])
    print("d AFFINE", json.dumps(T.jsonable(R["affine"]), indent=1), flush=True)

    # ---------------------------------------------------------------- e HOP BEHAVIOUR
    # on a 2-level sub-grid a clamped +-k step is a toggle-or-stay, so half the proposed
    # weight mutations are no-ops. Measure it rather than assert it.
    g5 = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    moved = proposed = 0
    dd_moved = dd_prop = 0
    for _ in range(120):
        h = G.mutate(g5, rng, p_add=0, p_prune=0, p_grow=0, p_shrink=0, p_affine=0,
                     p_delay=0.08, p_weight=0.25)
        m = g5["mask"] & h["mask"]
        moved += int((g5["weight"][m] != h["weight"][m]).sum())
        proposed += int(round(0.25 * m.sum()))
        dd_moved += int((g5["delay"][m] != h["delay"][m]).sum())
        dd_prop += int(round(0.08 * (m & ~G.PIN_DELAY).sum()))
        g5 = h
    R["hop"] = dict(weight_moves_per_proposed=moved / max(proposed, 1),
                    delay_moves_per_proposed=dd_moved / max(dd_prop, 1),
                    note="on a 2-level sub-grid a clamped step is a toggle-or-stay, so ~50 % "
                         "of proposed weight hops are no-ops and the EFFECTIVE p_weight is "
                         "about half its nominal value. Delays have 32 levels, so only the "
                         "two ends clamp and nearly every delay hop lands.",
                    ok=True)
    print("e HOP", json.dumps(T.jsonable(R["hop"]), indent=1), flush=True)

    # ---------------------------------------------------------------- f DEFAULTS UNCHANGED
    G.set_weight_levels(np.round(np.arange(-1.0, 1.0001, 0.2), 10))
    G.set_delay_levels(None)
    g6 = G.random_genome(rng, n_exc=8, n_inh=2)
    for _ in range(50):
        g6 = G.mutate(g6, rng)
    d6 = g6["delay"][g6["mask"]]
    R["defaults"] = dict(n_weight_levels=len(G.QUANT_LEVELS),
                         weights_on_11_grid=G.on_grid(g6)[0],
                         delay_grid_off=bool(G.DELAY_LEVELS is None),
                         even_delays_present=bool((d6 % 2 == 0).any()),
                         invariants=G.all_ok(g6),
                         ok=bool(len(G.QUANT_LEVELS) == 11 and G.on_grid(g6)[0]
                                 and (d6 % 2 == 0).any()
                                 and sum(G.all_ok(g6).values()) == 0))
    print("f DEFAULTS", json.dumps(T.jsonable(R["defaults"]), indent=1), flush=True)

    R["all_ok"] = bool(R["weights"]["ok"] and R["delays"]["ok"] and R["invariants"]["ok"]
                       and R["affine"]["ok"] and R["defaults"]["ok"])
    print(f"\nALL_OK={R['all_ok']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
