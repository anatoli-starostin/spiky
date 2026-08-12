"""exp012: validate weight quantisation before anything is run on it.

The grid is 11 symmetric levels at 0.2 spacing, SPLIT BY DALE: an excitatory source row may
only take a level from {0.0 .. 1.0}, an inhibitory row only from {-1.0 .. 0.0}. Quantisation
and Dale's law are therefore the same operation, and the checks below verify that a genome
cannot leave the grid by any route -- init, mutation, crossover, grow, or the seed embedding.

  a GRID        every weight on a legal level for its Dale half, at init and after mutation
  b INVARIANTS  the structural invariants are unaffected by quantisation
  c AFFINE      the calibration genes stay CONTINUOUS and near identity at init
  d MUTATION    the discrete hop has the intended +-1 / +-2 level distribution
  e SEED        an existing continuous genome snaps onto the grid, and how much it moves
  f CONTINUOUS  with the flag off, nothing changes -- weights are NOT on the grid and the
                Gaussian step is intact
"""
import argparse
import json
from collections import Counter

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load
from harness import LatencyEncoder
from tiny_evolve import load_ckpt as load_small_ckpt

BASE = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
        "exp012_tiny-direct-genome/")


def leader(pool, ewma):
    fin = np.where(np.isfinite(ewma))[0]
    return pool[int(fin[np.argmin(ewma[fin])])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fanout-cap", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    R = dict(levels=G.QUANT_LEVELS.tolist(), step=G.QUANT_STEP,
             positive_subgrid=G.QUANT_POS.tolist(), negative_subgrid=G.QUANT_NEG.tolist(),
             fanout_cap=a.fanout_cap)

    G.QUANTIZED = True
    G.FANOUT_CAP = a.fanout_cap

    # ---------------------------------------------------------------- a GRID
    off_grid = dale = 0
    seen = Counter()
    g = G.random_genome(rng, n_exc=8, n_inh=2)
    off_grid += G.on_grid(g)[1]
    for _ in range(a.trials):
        g = G.mutate(g, rng)
        off_grid += G.on_grid(g)[1]
        dale += G.dale_ok(g)[1]
        for v in np.round(g["weight"][g["mask"]], 10):
            seen[float(v)] += 1
    npos = G.N_IN + G.N_EXC_MAX
    exc_neg = int((g["weight"][:npos] < 0).sum())
    inh_pos = int((g["weight"][npos:] > 0).sum())
    R["grid"] = dict(genomes_checked=a.trials + 1, off_grid_weights=off_grid,
                     dale_violations=dale, exc_rows_negative=exc_neg,
                     inh_rows_positive=inh_pos,
                     distinct_levels_used=sorted(seen),
                     n_distinct=len(seen),
                     ok=bool(off_grid == 0 and dale == 0 and exc_neg == 0 and inh_pos == 0
                             and set(seen).issubset(set(G.QUANT_LEVELS.tolist()))))
    print("a GRID", json.dumps(T.jsonable(R["grid"]), indent=1), flush=True)

    # ---------------------------------------------------------------- b INVARIANTS
    tot = dict(dale=0, legal=0, pins=0, active=0, delays=0, range=0, fanout=0,
               in_to_inh=0, in_to_out=0, inh_to_out=0)
    grid2 = 0
    n = 0
    g = G.random_genome(rng, n_exc=8, n_inh=2)
    for _ in range(a.trials):
        g = G.mutate(g, rng)
        for k, v in G.all_ok(g).items():
            tot[k] += v
        grid2 += G.on_grid(g)[1]
        n += 1
    for _ in range(a.trials // 2):
        c = G.crossover(G.random_genome(rng), G.random_genome(rng), rng)
        for k, v in G.all_ok(c).items():
            tot[k] += v
        grid2 += G.on_grid(c)[1]
        n += 1
    # grow/shrink specifically, since they mint weights of their own
    g = G.random_genome(rng, n_exc=8, n_inh=2)
    for _ in range(50):
        G.grow(g, rng)
        grid2 += G.on_grid(g)[1]
        for k, v in G.all_ok(g).items():
            tot[k] += v
        n += 1
    for _ in range(30):
        G.shrink(g, rng)
        grid2 += G.on_grid(g)[1]
        n += 1
    R["invariants"] = dict(genomes_checked=n, violations=tot, off_grid=grid2,
                           ok=bool(sum(tot.values()) == 0 and grid2 == 0))
    print("b INVARIANTS", json.dumps(T.jsonable(R["invariants"]), indent=1), flush=True)

    # ---------------------------------------------------------------- c AFFINE
    g0 = G.random_genome(rng, n_exc=8, n_inh=2)
    aa0, bb0 = G.affine_of(g0)
    gm = g0
    for _ in range(100):
        gm = G.mutate(gm, rng)
    aa, bb = G.affine_of(gm)
    R["affine"] = dict(init_a=aa0.tolist(), init_b=bb0.tolist(),
                       identity_at_init=bool(np.all(aa0 == 1.0) and np.all(bb0 == 0.0)),
                       after_100_mutations_a=np.round(aa, 4).tolist(),
                       after_100_mutations_b=np.round(bb, 4).tolist(),
                       a_on_grid=bool(np.all(np.isin(np.round(aa, 10),
                                                     G.QUANT_LEVELS))),
                       still_continuous=bool(not np.all(np.isin(np.round(aa, 10),
                                                                G.QUANT_LEVELS))),
                       ok=True)
    R["affine"]["ok"] = bool(R["affine"]["identity_at_init"]
                             and R["affine"]["still_continuous"])
    print("c AFFINE", json.dumps(T.jsonable(R["affine"]), indent=1), flush=True)

    # ---------------------------------------------------------------- d MUTATION
    g = G.random_genome(rng, n_exc=8, n_inh=2)
    steps = Counter()
    for _ in range(400):
        h = G.mutate(g, rng, p_add=0, p_prune=0, p_delay=0, p_grow=0, p_shrink=0,
                     p_affine=0)
        d = np.round((h["weight"] - g["weight"])[g["mask"] & h["mask"]] / G.QUANT_STEP, 6)
        for v in d[d != 0]:
            steps[int(round(v))] += 1
        g = h
    tot_s = sum(steps.values())
    one = sum(v for k, v in steps.items() if abs(k) == 1) / max(tot_s, 1)
    two = sum(v for k, v in steps.items() if abs(k) == 2) / max(tot_s, 1)
    pos = sum(v for k, v in steps.items() if k > 0) / max(tot_s, 1)
    R["mutation"] = dict(n_steps=tot_s, hist={str(k): v for k, v in sorted(steps.items())},
                         frac_one_level=one, frac_two_level=two, frac_positive=pos,
                         want="80 % +-1, 20 % +-2, 50/50 sign; clamping at the rails "
                              "shifts these slightly",
                         ok=bool(0.70 < one < 0.90 and 0.10 < two < 0.30))
    print("d MUTATION", json.dumps(T.jsonable(R["mutation"]), indent=1), flush=True)

    # ---------------------------------------------------------------- e SEED
    g_lat = leader(*load_small_ckpt(BASE + "full_run_crossover_latinhib/ck_L0.npz")[:2])
    G.QUANTIZED = False
    g_cont = G.seed_from_small(g_lat)
    G.QUANTIZED = True
    g_quant = G.seed_from_small(g_lat)
    m = g_cont["mask"]
    dw = np.abs(g_quant["weight"] - g_cont["weight"])[m]
    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    mse_c = float(G.score(G.build([g_cont], device=a.device), Xv, Yv, enc)["mse"][0])
    mse_q = float(G.score(G.build([g_quant], device=a.device), Xv, Yv, enc)["mse"][0])
    R["seed"] = dict(on_grid=G.on_grid(g_quant)[0], n_synapses=int(m.sum()),
                     max_weight_shift=float(dw.max()), mean_weight_shift=float(dw.mean()),
                     mse_continuous=mse_c, mse_quantized=mse_q, mse_delta=mse_q - mse_c,
                     levels_used=sorted(set(np.round(g_quant["weight"][m], 10).tolist())),
                     ok=bool(G.on_grid(g_quant)[0]))
    print("e SEED", json.dumps(T.jsonable(R["seed"]), indent=1), flush=True)

    # ---------------------------------------------------------------- g FAN-OUT CAP
    # the cap has to hold from three directions: at init, under p_add, and under grow, which
    # is the only operator that can wire a brand-new neuron in both directions at once
    worst = 0
    n_syn = []
    g = G.random_genome(rng, n_exc=8, n_inh=2)
    worst = max(worst, int(G.hidden_fanout(g).max()))
    for _ in range(a.trials):
        g = G.mutate(g, rng)
        worst = max(worst, int(G.hidden_fanout(g).max()))
        n_syn.append(int(g["mask"].sum()))
    g_full = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    worst_full = int(G.hidden_fanout(g_full).max())
    for _ in range(a.trials):
        g_full = G.mutate(g_full, rng, p_grow=0, p_shrink=0)
        worst_full = max(worst_full, int(G.hidden_fanout(g_full).max()))
    g_grow = G.random_genome(rng, n_exc=2, n_inh=0)
    for _ in range(60):
        G.grow(g_grow, rng)
        worst = max(worst, int(G.hidden_fanout(g_grow).max()))
    # the ceiling the cap implies: inputs are uncapped, hidden rows are not
    n_in_edges = G.N_IN * G.N_EXC_MAX
    ceiling = n_in_edges + (G.N_EXC_MAX + G.N_INH_MAX) * a.fanout_cap
    R["fanout_cap"] = dict(
        cap=a.fanout_cap,
        worst_hidden_fanout_small_and_grow=worst,
        worst_hidden_fanout_full_capacity=worst_full,
        synapses_after_mutation_burst=dict(min=int(min(n_syn)), max=int(max(n_syn))),
        full_capacity_synapses_after_burst=int(g_full["mask"].sum()),
        implied_ceiling=ceiling,
        ceiling_note=(f"{n_in_edges} input->exc (uncapped) + 50 hidden x {a.fanout_cap} "
                      f"= {ceiling}; without the cap the legal-cell bound is 3420"),
        ok=bool(worst <= a.fanout_cap and worst_full <= a.fanout_cap))
    print("g FANOUT CAP", json.dumps(T.jsonable(R["fanout_cap"]), indent=1), flush=True)

    # ---------------------------------------------------------------- f CONTINUOUS UNCHANGED
    G.QUANTIZED = False
    G.FANOUT_CAP = None
    gc = G.random_genome(rng, n_exc=8, n_inh=2)
    before = gc["weight"].copy()
    for _ in range(50):
        gc = G.mutate(gc, rng)
    R["continuous_path"] = dict(
        off_grid_weights=G.on_grid(gc)[1],
        weights_moved=bool(not np.array_equal(before, gc["weight"])),
        invariants=G.all_ok(gc),
        ok=bool(G.on_grid(gc)[1] > 0 and sum(G.all_ok(gc).values()) == 0))
    print("f CONTINUOUS", json.dumps(T.jsonable(R["continuous_path"]), indent=1), flush=True)

    R["all_ok"] = bool(R["grid"]["ok"] and R["invariants"]["ok"] and R["affine"]["ok"]
                       and R["mutation"]["ok"] and R["seed"]["ok"]
                       and R["fanout_cap"]["ok"] and R["continuous_path"]["ok"])
    print(f"\nALL_OK={R['all_ok']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
