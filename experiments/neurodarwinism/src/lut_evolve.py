"""exp011: the chapter's steady-state evolutionary loop, over LUT hyperparameters.

Same loop shape as `steady_state.py` -- K members in a pool, an EWMA of each member's score,
cull the worst M past a grace period, refill from fitness-weighted tournament-of-2 parents,
track lineages, checkpoint every round. What changes is the genome and the inner learning step:

    steady_state.py   genome = ~100k synapses          inner step = STDP
    lut_evolve.py     genome = LUT hyperparameters     inner step = Adam on minibatches

WHY NOT IMPORT steady_state's MUTATION AND BUILD. Its `mutate_structural` / `build_pool` /
`readback` are all typed to the synapse-array genome -- they add and prune synapses, enforce
Dale's law, and read weights back off a SpikingNet. None of that has a meaning for a 9-key
hyperparameter dict. The reusable part of that file is the LOOP, which is reproduced here
faithfully (same EWMA, same cull rule, same grace period, same tournament, same lineage
bookkeeping) rather than imported through a type it does not fit.

FITNESS = -held_out_MSE - lambda * param_count, so selection is pushed toward the smallest LUT
that still fits. lambda defaults to a value that makes the size term comparable to a meaningful
MSE difference at the teacher's own size (see --size-penalty). The RAW (held_out_mse,
param_count) of every member is logged every round regardless, so the fit-vs-size Pareto front
is readable straight out of the history and does not depend on the lambda that was chosen.

STOCHASTICITY. A member is re-trained from scratch every round with a different training seed,
so its score is a noisy sample of "what this architecture reaches". The EWMA is what turns that
into a usable signal -- exactly the role it plays in the SPNet loop.

    sbox python lut_evolve.py --pool 8 --rounds 5 --steps 500        # smoke test
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lut_backprop as lb                                        # noqa: E402

OUT = os.path.abspath(os.environ.get("ND_OUT", os.path.join(HERE, "results")))


# ----------------------------------------------------------------- genome
def seed_genome(rng, evolve_heads=False, forward_modes=("hard",)):
    g = dict(lb.DEFAULT_GENOME)
    g["n_anchor_pairs"] = int(rng.integers(lb.NAP_RANGE[0], lb.NAP_RANGE[1] + 1))
    g["tables_per_head"] = int(rng.integers(lb.TPH_RANGE[0], lb.TPH_RANGE[1] + 1))
    g["n_heads"] = int(rng.integers(*lb.HEADS_RANGE)) if evolve_heads else 1
    g["forward_mode"] = str(rng.choice(list(forward_modes)))
    g["lr"] = float(np.exp(rng.uniform(math.log(lb.LR_RANGE[0]), math.log(lb.LR_RANGE[1]))))
    g["anchor_seed"] = int(rng.integers(0, 100000))
    return g


def mutate(g, rng, evolve_heads=False, forward_modes=("hard",), p=0.5):
    """Perturb the hyperparameters. Every knob is bounded and every result is buildable.

    NAP and tables_per_head move MULTIPLICATIVELY (+-1 on NAP is a factor of 2 in rows; tph
    takes a log-normal kick), because capacity here is exponential in NAP and the search would
    otherwise crawl at the large end and thrash at the small one.
    """
    h = dict(g)
    if rng.random() < p:
        h["n_anchor_pairs"] = int(np.clip(h["n_anchor_pairs"] + rng.choice([-1, 1]),
                                          *lb.NAP_RANGE))
    if rng.random() < p:
        h["tables_per_head"] = int(np.clip(
            round(h["tables_per_head"] * math.exp(rng.normal(0, 0.5))), *lb.TPH_RANGE))
    if evolve_heads and rng.random() < 0.25:
        h["n_heads"] = int(np.clip(h["n_heads"] + rng.choice([-1, 1]), *lb.HEADS_RANGE))
    if rng.random() < p:
        h["lr"] = float(np.clip(h["lr"] * math.exp(rng.normal(0, 0.4)), *lb.LR_RANGE))
    if len(forward_modes) > 1 and rng.random() < 0.2:
        h["forward_mode"] = str(rng.choice(list(forward_modes)))
    if rng.random() < 0.2:
        h["anchor_seed"] = int(rng.integers(0, 100000))
    return h


# ----------------------------------------------------------------- fitness
def fitness(mse, params, lam):
    """-MSE - lambda * params. Selection maximises, as everywhere else in the chapter."""
    return -mse - lam * params


def pareto(front):
    """Non-dominated (params, mse) points, smallest first. The number the brief actually wants:
    'the minimal config that reaches MSE threshold X' is read straight off this."""
    pts = sorted(front, key=lambda r: (r["params"], r["mse"]))
    out, best = [], float("inf")
    for r in pts:
        if r["mse"] < best - 1e-12:
            out.append(r)
            best = r["mse"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--cull", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--grace", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500, help="Adam steps per candidate per round")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size-penalty", type=float, default=2e-7,
                    help="lambda in -MSE - lambda*params. The default charges the teacher's "
                         "12,288 params 0.0025 of MSE, which is ~8%% of the MSE a config that "
                         "size reaches -- enough to break ties toward smaller, not enough to "
                         "dominate fit. The Pareto front is logged raw and is lambda-free.")
    ap.add_argument("--evolve-heads", action="store_true",
                    help="let n_heads vary. Off by default: under a summed readout it is the "
                         "same capacity axis as tables_per_head (see lut_backprop's docstring)")
    ap.add_argument("--forward-modes", nargs="+", default=["hard"],
                    choices=["hard", "hybrid_smooth"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    out_dir = a.out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    Xtr, Ytr, Xte, Yte = lb.to_device(a.seed, a.n_val, dev)
    base = lb.baselines(Ytr, Yte)
    print(f"exp011 evolve: K={a.pool}, {a.rounds} rounds, {a.steps} Adam steps/candidate, "
          f"lambda {a.size_penalty:g}, dev {dev}")
    print(f"  constant-predictor held-out MSE {base['constant_predictor_mse']:.5f}, "
          f"target sd {base['target_sd']:.4f}")

    M = max(1, int(a.cull * a.pool))
    genomes = [seed_genome(np.random.default_rng(a.seed * 100 + i), a.evolve_heads,
                           a.forward_modes) for i in range(a.pool)]
    ewma = np.full(a.pool, np.nan)
    age = np.zeros(a.pool, int)
    lineage = np.arange(a.pool)
    hist, seen, t0 = [], [], time.time()

    for rnd in range(a.rounds):
        fit = np.zeros(a.pool)
        mses = np.zeros(a.pool)
        pars = np.zeros(a.pool, dtype=np.int64)
        for i, g in enumerate(genomes):
            # a FRESH training seed per (round, member): the score is a noisy sample of what
            # the architecture reaches, and the EWMA is what averages it
            r = lb.train_eval(g, Xtr, Ytr, Xte, Yte, a.steps, a.batch,
                              seed=a.seed * 100003 + rnd * 97 + i, device=dev)
            mses[i], pars[i] = r["heldout_mse"], r["params"]
            fit[i] = fitness(mses[i], pars[i], a.size_penalty)
            seen.append(dict(rnd=rnd, member=i, mse=float(mses[i]), params=int(pars[i]),
                             genome=r["genome"]))
        ewma = np.where(np.isnan(ewma), fit, (1 - a.alpha) * ewma + a.alpha * fit)
        age += 1

        eligible = np.nonzero(age > a.grace)[0]
        if eligible.size >= M:
            worst = eligible[np.argsort(ewma[eligible])[:M]]
            surv = np.setdiff1d(np.arange(a.pool), worst)
            for slot in worst:
                c1, c2 = rng.choice(surv, 2, replace=False)
                par = c1 if ewma[c1] >= ewma[c2] else c2
                genomes[slot] = mutate(genomes[par], rng, a.evolve_heads, a.forward_modes)
                ewma[slot] = ewma[par]
                age[slot] = 0
                lineage[slot] = lineage[par]

        bi = int(np.nanargmax(ewma))
        pf = pareto(seen)
        rec = dict(rnd=rnd, best=float(np.nanmax(ewma)), mean=float(np.nanmean(ewma)),
                   batch_best=float(fit.max()),
                   min_mse=float(mses.min()), min_params=int(pars.min()),
                   median_params=int(np.median(pars)),
                   n_lineages=int(np.unique(lineage).size),
                   wall=round(time.time() - t0, 1),
                   fitness_vec=[round(float(v), 6) for v in fit],
                   mse_vec=[round(float(v), 6) for v in mses],
                   params_vec=[int(v) for v in pars],
                   pareto=[dict(params=p["params"], mse=round(p["mse"], 6),
                                nap=p["genome"]["n_anchor_pairs"],
                                tph=p["genome"]["tables_per_head"]) for p in pf])
        hist.append(rec)
        print(f"  round {rnd:3d}  best fitness {np.nanmax(ewma):+.5f}  "
              f"min MSE {mses.min():.5f}  min params {pars.min():,}  "
              f"median params {int(np.median(pars)):,}  "
              f"pareto {len(pf)} pts  {rec['wall']:.0f}s", flush=True)
        json.dump(hist, open(os.path.join(out_dir, f"lut_evolve{a.tag}.json"), "w"), indent=1)
        if a.ckpt:
            json.dump(dict(genomes=genomes, ewma=ewma.tolist(), age=age.tolist(),
                           lineage=lineage.tolist(), next_rnd=rnd + 1, seen=seen),
                      open(a.ckpt, "w"))

    print(f"\nBEST member {bi}: fitness {ewma[bi]:+.5f}")
    print(f"  {lb.genome_str(genomes[bi])}")
    print(f"\nPARETO FRONT (fit vs size), {len(pareto(seen))} points, "
          f"lambda-free -- read the minimal config for any MSE threshold off this:")
    print(f"  {'params':>10} {'held-out MSE':>13} {'vs constant':>12}  config")
    for p in pareto(seen):
        g = p["genome"]
        print(f"  {p['params']:10,} {p['mse']:13.5f} "
              f"{p['mse'] / base['constant_predictor_mse']:11.4f}x  "
              f"NAP {g['n_anchor_pairs']:2d} x tph {g['tables_per_head']:3d} "
              f"x heads {g['n_heads']}  lr {g['lr']:.4g}")
    json.dump(dict(baselines=base, history=hist, seen=seen,
                   pareto=pareto(seen), best_genome=genomes[bi]),
              open(os.path.join(out_dir, f"lut_evolve{a.tag}_final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
