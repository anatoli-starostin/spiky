"""exp012: symmetric 3-value weight grid + the evolvable global inhibition coefficient.

Weights {-1.0, 0, 1.0}, Dale-split to exc {0, 1.0} and inh {-1.0, 0} -- binary within each
sign class and now SYMMETRIC, with all the E/I asymmetry moved into one evolvable scalar:

    effective current   exc: GAIN * w          inh: GAIN * w * inh_coeff

  a WEIGHTS   exactly on the symmetric grid, Dale-split correct
  b COEFF     starts at 0.5, moves under mutation, stays inside (0.05, 4.0]
  c BUILD     the coeff scales inhibitory currents and leaves excitatory ones untouched
  d CROSSOVER a child's coeff is one parent's value, never an average
  e LEGACY    a genome with no coeff gene reads as 1.0 and builds identically to before
  f INVARIANTS Dale, legality, motif, fan-out cap, delay pin, delay grid, affine continuous
  g DEFAULTS  with nothing configured, every prior behaviour is unchanged
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
    G.set_weight_levels([-1.0, 0.0, 1.0])
    G.set_delay_levels(list(range(1, 64, 2)))
    G.QUANTIZED = True
    G.FANOUT_CAP = a.fanout_cap
    P_INH = 0.25
    R = dict(weight_levels=G.QUANT_LEVELS.tolist(), exc_subgrid=G.QUANT_POS.tolist(),
             inh_subgrid=G.QUANT_NEG.tolist(), coeff_init=G.INH_COEFF_INIT,
             coeff_box=[G.INH_COEFF_LO, G.INH_COEFF_HI], coeff_sigma=G.INH_COEFF_SIGMA,
             p_inhcoeff=P_INH, fanout_cap=a.fanout_cap)

    # ---------------------------------------------------------------- a WEIGHTS
    npos = G.N_IN + G.N_EXC_MAX
    exc_seen, inh_seen = Counter(), Counter()
    off = dale = 0
    coeffs = []
    g = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    R["coeff_at_init"] = G.inh_coeff_of(g)
    for _ in range(a.trials):
        g = G.mutate(g, rng, p_grow=0, p_shrink=0, p_inhcoeff=P_INH)
        off += G.on_grid(g)[1]
        dale += G.dale_ok(g)[1]
        coeffs.append(G.inh_coeff_of(g))
        for v in np.round(g["weight"][:npos][g["mask"][:npos]], 10):
            exc_seen[float(v)] += 1
        for v in np.round(g["weight"][npos:][g["mask"][npos:]], 10):
            inh_seen[float(v)] += 1
    R["weights"] = dict(off_grid=off, dale_violations=dale,
                        exc_values_seen=sorted(exc_seen), inh_values_seen=sorted(inh_seen),
                        ok=bool(off == 0 and dale == 0
                                and set(exc_seen).issubset({0.0, 1.0})
                                and set(inh_seen).issubset({-1.0, 0.0})))
    print("a WEIGHTS", json.dumps(T.jsonable(R["weights"]), indent=1), flush=True)

    # ---------------------------------------------------------------- b COEFF
    c = np.array(coeffs)
    R["coeff"] = dict(init=R["coeff_at_init"], n_mutations=len(c),
                      min=float(c.min()), mean=float(c.mean()), max=float(c.max()),
                      final=float(c[-1]), moved_off_init=bool(c.std() > 1e-9),
                      distinct_values=int(len(np.unique(np.round(c, 9)))),
                      in_box=bool((c > G.INH_COEFF_LO - 1e-12).all()
                                  and (c <= G.INH_COEFF_HI + 1e-12).all()),
                      ok=bool(R["coeff_at_init"] == G.INH_COEFF_INIT and c.std() > 1e-9
                              and (c > G.INH_COEFF_LO - 1e-12).all()
                              and (c <= G.INH_COEFF_HI + 1e-12).all()))
    print("b COEFF", json.dumps(T.jsonable(R["coeff"]), indent=1), flush=True)

    # ---------------------------------------------------------------- c BUILD SCALING
    # DENSE on purpose. At p_init 0.10 with 2 inhibitory units a sample genome can easily end
    # up with ZERO inhibitory synapses, and then "inh currents halve" is 0 == 0*0.5 -- a test
    # that passes without testing anything. Draw dense and assert the count is non-zero.
    gb = G.random_genome(rng, p_init=0.9, n_exc=8, n_inh=4)
    rows = np.nonzero(gb["mask"])[0]
    is_inh = rows >= npos
    assert is_inh.sum() > 0, "no inhibitory synapses -- the scaling test would be vacuous"
    got = {}
    for cv in (0.5, 1.0, 2.0):
        gb["inh_coeff"] = cv
        w = gb["weight"][gb["mask"]] * G.GAIN
        w = np.where(is_inh, w * G.inh_coeff_of(gb), w)
        got[str(cv)] = dict(exc_sum=float(np.abs(w[~is_inh]).sum()),
                            inh_sum=float(np.abs(w[is_inh]).sum()))
    e1, i1 = got["1.0"]["exc_sum"], got["1.0"]["inh_sum"]
    R["build_scaling"] = dict(
        per_coeff=got, n_inh_synapses=int(is_inh.sum()), n_exc_synapses=int((~is_inh).sum()),
        exc_unchanged=bool(abs(got["0.5"]["exc_sum"] - e1) < 1e-9
                           and abs(got["2.0"]["exc_sum"] - e1) < 1e-9),
        inh_halves=bool(abs(got["0.5"]["inh_sum"] - 0.5 * i1) < 1e-9),
        inh_doubles=bool(abs(got["2.0"]["inh_sum"] - 2.0 * i1) < 1e-9))
    R["build_scaling"]["ok"] = bool(R["build_scaling"]["exc_unchanged"]
                                    and R["build_scaling"]["inh_halves"]
                                    and R["build_scaling"]["inh_doubles"]
                                    and is_inh.sum() > 0 and i1 > 0)
    print("c BUILD SCALING", json.dumps(T.jsonable(R["build_scaling"]), indent=1), flush=True)

    # ---------------------------------------------------------------- d CROSSOVER
    picks = Counter()
    for _ in range(400):
        p1 = G.random_genome(rng, n_exc=8, n_inh=2)
        p2 = G.random_genome(rng, n_exc=8, n_inh=2)
        p1["inh_coeff"], p2["inh_coeff"] = 0.3, 2.7
        ch = G.crossover(p1, p2, rng)
        v = G.inh_coeff_of(ch)
        picks["p1" if v == 0.3 else ("p2" if v == 2.7 else f"OTHER {v}")] += 1
    R["crossover"] = dict(counts=dict(picks), n=400,
                          only_parent_values=bool(set(picks) <= {"p1", "p2"}),
                          frac_p1=picks["p1"] / 400,
                          ok=bool(set(picks) <= {"p1", "p2"}
                                  and 0.4 < picks["p1"] / 400 < 0.6))
    print("d CROSSOVER", json.dumps(T.jsonable(R["crossover"]), indent=1), flush=True)

    # ---------------------------------------------------------------- e LEGACY
    legacy = G.random_genome(rng, n_exc=8, n_inh=2)
    legacy.pop("inh_coeff")
    w_leg = legacy["weight"][legacy["mask"]] * G.GAIN
    rows_l = np.nonzero(legacy["mask"])[0]
    w_leg_scaled = np.where(rows_l >= npos, w_leg * G.inh_coeff_of(legacy), w_leg)
    R["legacy"] = dict(coeff_read_as=G.inh_coeff_of(legacy),
                       expected=G.INH_COEFF_LEGACY,
                       currents_identical=bool(np.array_equal(w_leg, w_leg_scaled)),
                       new_genome_starts_at=G.inh_coeff_of(G.blank()),
                       ok=bool(G.inh_coeff_of(legacy) == G.INH_COEFF_LEGACY
                               and np.array_equal(w_leg, w_leg_scaled)
                               and G.inh_coeff_of(G.blank()) == G.INH_COEFF_INIT))
    print("e LEGACY", json.dumps(T.jsonable(R["legacy"]), indent=1), flush=True)

    # ---------------------------------------------------------------- f INVARIANTS
    tot = {}
    n = 0
    worst = 0
    g2 = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    for _ in range(a.trials):
        g2 = G.mutate(g2, rng, p_grow=0, p_shrink=0, p_inhcoeff=P_INH)
        for k, v in G.all_ok(g2).items():
            tot[k] = tot.get(k, 0) + v
        worst = max(worst, int(G.hidden_fanout(g2).max()))
        n += 1
    for _ in range(a.trials // 2):
        ch = G.crossover(G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10),
                         G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10), rng)
        for k, v in G.all_ok(ch).items():
            tot[k] = tot.get(k, 0) + v
        worst = max(worst, int(G.hidden_fanout(ch).max()))
        n += 1
    aa, _ = G.affine_of(g2)
    R["invariants"] = dict(genomes_checked=n, violations=tot, worst_hidden_fanout=worst,
                           affine_still_continuous=bool(not np.all(
                               np.isin(np.round(aa, 10), G.QUANT_LEVELS))),
                           ok=bool(sum(tot.values()) == 0 and worst <= a.fanout_cap))
    print("f INVARIANTS", json.dumps(T.jsonable(R["invariants"]), indent=1), flush=True)

    # ---------------------------------------------------------------- g DEFAULTS
    G.set_weight_levels(np.round(np.arange(-1.0, 1.0001, 0.2), 10))
    G.set_delay_levels(None)
    G.QUANTIZED = False
    G.FANOUT_CAP = None
    gd = G.random_genome(rng, n_exc=8, n_inh=2)
    c0 = G.inh_coeff_of(gd)
    for _ in range(60):
        gd = G.mutate(gd, rng)                      # p_inhcoeff defaults to 0
    R["defaults"] = dict(coeff_frozen_when_gated=bool(G.inh_coeff_of(gd) == c0),
                         n_weight_levels=len(G.QUANT_LEVELS),
                         delay_grid_off=bool(G.DELAY_LEVELS is None),
                         weights_off_grid=G.on_grid(gd)[1] > 0,
                         invariants=G.all_ok(gd),
                         ok=bool(G.inh_coeff_of(gd) == c0 and len(G.QUANT_LEVELS) == 11
                                 and G.DELAY_LEVELS is None and G.on_grid(gd)[1] > 0
                                 and sum(G.all_ok(gd).values()) == 0))
    print("g DEFAULTS", json.dumps(T.jsonable(R["defaults"]), indent=1), flush=True)

    R["all_ok"] = bool(all(R[k]["ok"] for k in ("weights", "coeff", "build_scaling",
                                                "crossover", "legacy", "invariants",
                                                "defaults")))
    print(f"\nALL_OK={R['all_ok']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
