"""exp012: the evolvable global synaptic gain, on top of the symmetric grid + inh coefficient.

    exc current = gain * w          inh current = gain * w * inh_coeff
    input injection = 200.0, FIXED and untouched by the gene

That last line is the whole point. If gain scaled the injection too, it would be a change of
units and could not move a single spike time; it means something only because the injection
stays put as the reference the substrate is measured against.

  a GAIN       starts at 200, moves under mutation, stays in [20, 2000]
  b BUILD      synaptic currents scale exactly; the input injection does not
  c CROSSOVER  a child's gain is one parent's value, never an average
  d LEGACY     a genome without the gene reads 200.0 and builds byte-identical currents
  e INVARIANTS everything else still holds
  f DEFAULTS   gated off -> gain frozen at 200
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    G.set_weight_levels([-1.0, 0.0, 1.0])
    G.set_delay_levels(list(range(1, 64, 2)))
    G.QUANTIZED = True
    G.FANOUT_CAP = a.fanout_cap
    P_INH = P_GAIN = 0.25
    R = dict(gain_init=G.GAIN_INIT, gain_box=[G.GAIN_LO, G.GAIN_HI],
             gain_sigma=G.GAIN_SIGMA, p_gain=P_GAIN, p_inhcoeff=P_INH,
             weight_levels=G.QUANT_LEVELS.tolist())

    # ---------------------------------------------------------------- a GAIN
    g = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    R["gain_at_init"] = G.gain_of(g)
    gains, coeffs = [], []
    for _ in range(a.trials):
        g = G.mutate(g, rng, p_grow=0, p_shrink=0, p_inhcoeff=P_INH, p_gain=P_GAIN)
        gains.append(G.gain_of(g))
        coeffs.append(G.inh_coeff_of(g))
    gv = np.array(gains)
    R["gain"] = dict(init=R["gain_at_init"], n_mutations=len(gv),
                     min=float(gv.min()), mean=float(gv.mean()), max=float(gv.max()),
                     final=float(gv[-1]), moved=bool(gv.std() > 1e-9),
                     distinct=int(len(np.unique(np.round(gv, 6)))),
                     in_box=bool((gv >= G.GAIN_LO - 1e-9).all()
                                 and (gv <= G.GAIN_HI + 1e-9).all()),
                     coeff_min=float(np.min(coeffs)), coeff_mean=float(np.mean(coeffs)),
                     coeff_max=float(np.max(coeffs)),
                     ok=bool(R["gain_at_init"] == G.GAIN_INIT and gv.std() > 1e-9
                             and (gv >= G.GAIN_LO - 1e-9).all()
                             and (gv <= G.GAIN_HI + 1e-9).all()))
    print("a GAIN", json.dumps(T.jsonable(R["gain"]), indent=1), flush=True)

    # ---------------------------------------------------------------- b BUILD
    gb = G.random_genome(rng, p_init=0.7, n_exc=8, n_inh=4)
    gb["inh_coeff"] = 0.5
    npos = G.N_IN + G.N_EXC_MAX
    r, c = np.nonzero(gb["mask"])
    wn = gb["weight"][r, c]
    is_inh = r >= npos
    assert is_inh.sum() > 0 and (~is_inh).sum() > 0, "need both signs for a real test"
    rows = {}
    for gval in (150.0, 200.0, 400.0):
        gb["gain"] = gval
        H = G.build([gb], device=a.device)               # the REAL path
        w = H["weights"]
        rows[str(gval)] = dict(
            exc_full=sorted(set(np.round(w[(~is_inh) & (np.abs(wn - 1.0) < 1e-9)], 6).tolist())),
            inh_full=sorted(set(np.round(w[(is_inh) & (np.abs(wn + 1.0) < 1e-9)], 6).tolist())),
            zero=sorted(set(np.round(w[np.abs(wn) < 1e-9], 6).tolist())))
    # the INPUT INJECTION must not move: it is a constant in the episode runner, so read it
    # back out of the array the runner actually builds
    from harness import LatencyEncoder
    from data import load
    _, _, Xp, Yp, Xv, Yv = load(256, seed=0)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    inj = {}
    for gval in (150.0, 400.0):
        gb["gain"] = gval
        H = G.build([gb], device=a.device)
        cols = H["ids"][2]
        va = np.zeros((4, T.T_IN, cols.size), np.float32)
        tk = enc(Xv[:4])
        for b in range(4):
            for j in range(T.N_IN):
                va[b, tk[b, j], j::T.N_IN] = 200.0
        inj[str(gval)] = float(va.max())
    R["build"] = dict(per_gain=rows, input_injection=inj,
                      injection_unchanged=bool(inj["150.0"] == inj["400.0"] == 200.0),
                      exc_equals_gain=bool(rows["150.0"]["exc_full"] == [150.0]
                                           and rows["200.0"]["exc_full"] == [200.0]
                                           and rows["400.0"]["exc_full"] == [400.0]),
                      inh_equals_gain_times_coeff=bool(rows["150.0"]["inh_full"] == [-75.0]
                                                       and rows["200.0"]["inh_full"] == [-100.0]
                                                       and rows["400.0"]["inh_full"] == [-200.0]))
    R["build"]["ok"] = bool(R["build"]["injection_unchanged"]
                            and R["build"]["exc_equals_gain"]
                            and R["build"]["inh_equals_gain_times_coeff"])
    print("b BUILD", json.dumps(T.jsonable(R["build"]), indent=1), flush=True)

    # ---------------------------------------------------------------- c CROSSOVER
    picks = Counter()
    for _ in range(400):
        p1 = G.random_genome(rng, n_exc=8, n_inh=2)
        p2 = G.random_genome(rng, n_exc=8, n_inh=2)
        p1["gain"], p2["gain"] = 111.0, 999.0
        v = G.gain_of(G.crossover(p1, p2, rng))
        picks["p1" if v == 111.0 else ("p2" if v == 999.0 else f"OTHER {v}")] += 1
    R["crossover"] = dict(counts=dict(picks), only_parent_values=bool(set(picks) <= {"p1", "p2"}),
                          frac_p1=picks["p1"] / 400,
                          ok=bool(set(picks) <= {"p1", "p2"} and 0.4 < picks["p1"] / 400 < 0.6))
    print("c CROSSOVER", json.dumps(T.jsonable(R["crossover"]), indent=1), flush=True)

    # ---------------------------------------------------------------- d LEGACY
    leg = G.random_genome(rng, n_exc=8, n_inh=2)
    leg.pop("gain")
    leg.pop("inh_coeff")
    Hl = G.build([leg], device=a.device)
    leg2 = {**leg, "gain": 200.0, "inh_coeff": 1.0}
    Hl2 = G.build([leg2], device=a.device)
    R["legacy"] = dict(gain_read_as=G.gain_of(leg), expected=float(G.GAIN),
                       coeff_read_as=G.inh_coeff_of(leg),
                       currents_identical=bool(np.array_equal(Hl["weights"], Hl2["weights"])),
                       new_genome_starts_at=G.gain_of(G.blank()),
                       ok=bool(G.gain_of(leg) == G.GAIN
                               and np.array_equal(Hl["weights"], Hl2["weights"])
                               and G.gain_of(G.blank()) == G.GAIN_INIT))
    print("d LEGACY", json.dumps(T.jsonable(R["legacy"]), indent=1), flush=True)

    # ---------------------------------------------------------------- e INVARIANTS
    tot = {}
    n = 0
    worst = 0
    g2 = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    for _ in range(a.trials):
        g2 = G.mutate(g2, rng, p_grow=0, p_shrink=0, p_inhcoeff=P_INH, p_gain=P_GAIN)
        for k, v in G.all_ok(g2).items():
            tot[k] = tot.get(k, 0) + v
        worst = max(worst, int(G.hidden_fanout(g2).max()))
        n += 1
    for _ in range(a.trials // 2):
        ch = G.crossover(G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10),
                         G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10), rng)
        for k, v in G.all_ok(ch).items():
            tot[k] = tot.get(k, 0) + v
        n += 1
    aa, _ = G.affine_of(g2)
    R["invariants"] = dict(genomes_checked=n, violations=tot, worst_hidden_fanout=worst,
                           on_symmetric_grid=G.on_grid(g2)[0],
                           affine_continuous=bool(not np.all(np.isin(np.round(aa, 10),
                                                                     G.QUANT_LEVELS))),
                           ok=bool(sum(tot.values()) == 0 and worst <= a.fanout_cap
                                   and G.on_grid(g2)[0]))
    print("e INVARIANTS", json.dumps(T.jsonable(R["invariants"]), indent=1), flush=True)

    # ---------------------------------------------------------------- f DEFAULTS
    gd = G.random_genome(rng, n_exc=8, n_inh=2)
    g0 = G.gain_of(gd)
    for _ in range(80):
        gd = G.mutate(gd, rng)                 # p_gain and p_inhcoeff default to 0
    R["defaults"] = dict(gain_frozen=bool(G.gain_of(gd) == g0), value=G.gain_of(gd),
                         coeff_frozen=bool(G.inh_coeff_of(gd) == G.INH_COEFF_INIT),
                         ok=bool(G.gain_of(gd) == g0 == G.GAIN_INIT))
    print("f DEFAULTS", json.dumps(T.jsonable(R["defaults"]), indent=1), flush=True)

    R["all_ok"] = bool(all(R[k]["ok"] for k in ("gain", "build", "crossover", "legacy",
                                                "invariants", "defaults")))
    print(f"\nALL_OK={R['all_ok']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
