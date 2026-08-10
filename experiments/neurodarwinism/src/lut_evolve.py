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


# ----------------------------------------------------------------- literal anchor pairs
def _fresh_pair(rng, used, input_dim=lb.N_IN, tries=32):
    """A canonical (a<b) pair not already present in `used` (a set of a*input_dim+b keys)."""
    for _ in range(tries):
        a, b = rng.integers(0, input_dim, 2)
        if a == b:
            continue
        a, b = (a, b) if a < b else (b, a)
        if a * input_dim + b not in used:
            return int(a), int(b)
    # exhaustive fallback -- with C(17,2)=136 pairs and NAP <= 12 this is essentially never hit,
    # but a silent duplicate would violate within-table distinctness, so never guess
    pool = lb.canonical_pool(input_dim)
    free = [p for p in pool if p[0] * input_dim + p[1] not in used]
    if not free:
        return None
    p = free[int(rng.integers(0, len(free)))]
    return int(p[0]), int(p[1])


def resize_pairs(pairs, want_tables, want_nap, rng, input_dim=lb.N_IN):
    """Reshape an evolved pair set after NAP or tables_per_head mutated.

    Growing PRESERVES what evolution already found and only fills the new slots; shrinking
    truncates. Redrawing everything on a shape change would throw away the anchoring search
    every time a size gene moved, which is most mutations.
    """
    cur = np.asarray(pairs, np.int64)
    n_t, nap = cur.shape[0], cur.shape[1]
    if want_nap < nap:
        cur = cur[:, :want_nap]
    elif want_nap > nap:
        add = np.zeros((n_t, want_nap - nap, 2), np.int64)
        for t in range(n_t):
            used = {int(a) * input_dim + int(b) for a, b in cur[t]}
            for j in range(want_nap - nap):
                p = _fresh_pair(rng, used, input_dim)
                add[t, j] = p
                used.add(p[0] * input_dim + p[1])
        cur = np.concatenate([cur, add], 1)
    if want_tables < cur.shape[0]:
        cur = cur[:want_tables]
    elif want_tables > cur.shape[0]:
        extra = want_tables - cur.shape[0]
        new = np.zeros((extra, want_nap, 2), np.int64)
        for t in range(extra):
            used = set()
            for j in range(want_nap):
                p = _fresh_pair(rng, used, input_dim)
                new[t, j] = p
                used.add(p[0] * input_dim + p[1])
        cur = np.concatenate([cur, new], 0)
    return cur


def mutate_pairs(pairs, rng, p_pair=0.03, input_dim=lb.N_IN):
    """Edit INDIVIDUAL anchor pairs -- the gradient-free half of the split.

    Two operators, chosen 50/50 per selected pair:
      RESAMPLE  replace the pair with a fresh canonical one not already in that table
      NUDGE     move ONE endpoint to a different input dimension, re-canonicalising so a < b

    NUDGE is the local move (one dimension changes, the comparison stays "nearby"); RESAMPLE is
    the jump. Both preserve a<b, the index range, and within-table distinctness, which is what
    pairs_valid() checks and what both of the module's own samplers guarantee.

    p_pair is per PAIR, so the expected number of edits scales with n_tables * NAP -- a 32x6
    genome gets ~6 edits per mutation, a 128x12 genome ~46. That keeps the edit RATE per
    anchor constant instead of making big genomes effectively immutable.
    """
    p = np.array(pairs, np.int64, copy=True)
    n_t, nap = p.shape[0], p.shape[1]
    n_edit = 0
    for t in range(n_t):
        used = {int(a) * input_dim + int(b) for a, b in p[t]}
        for j in range(nap):
            if rng.random() >= p_pair:
                continue
            a0, b0 = int(p[t, j, 0]), int(p[t, j, 1])
            used.discard(a0 * input_dim + b0)
            if rng.random() < 0.5:
                new = _fresh_pair(rng, used, input_dim)
            else:
                keep = b0 if rng.random() < 0.5 else a0
                new = None
                for _ in range(32):
                    o = int(rng.integers(0, input_dim))
                    if o == keep:
                        continue
                    cand = (min(o, keep), max(o, keep))
                    if cand[0] * input_dim + cand[1] not in used:
                        new = cand
                        break
                if new is None:
                    new = _fresh_pair(rng, used, input_dim)
            if new is None:                       # nothing free: keep the original
                used.add(a0 * input_dim + b0)
                continue
            p[t, j] = new
            used.add(new[0] * input_dim + new[1])
            n_edit += 1
    return p, n_edit


# ----------------------------------------------------------------- genome
def seed_genome(rng, evolve_heads=False, evolve_pairs=True):
    """The forward mode is NOT part of the genome: exp011 is hard-forward only, and the
    surrogate backward is what makes that trainable (see lut_backprop.FORWARD_MODE)."""
    g = dict(lb.DEFAULT_GENOME)
    g["n_anchor_pairs"] = int(rng.integers(lb.NAP_RANGE[0], lb.NAP_RANGE[1] + 1))
    g["tables_per_head"] = int(rng.integers(lb.TPH_RANGE[0], lb.TPH_RANGE[1] + 1))
    g["n_heads"] = int(rng.integers(*lb.HEADS_RANGE)) if evolve_heads else 1
    g["lr"] = float(np.exp(rng.uniform(math.log(lb.LR_RANGE[0]), math.log(lb.LR_RANGE[1]))))
    # THE ANCHORING AXIS: the drawing RULE (policy) and the DRAW (seed). See
    # lut_backprop.ANCHOR_POLICIES for why the supported set is exactly these two.
    g["anchor_policy"] = str(rng.choice(lb.ANCHOR_POLICIES))
    g["anchor_seed"] = int(rng.integers(0, 100000))
    if evolve_pairs:
        # seeded by the module's OWN sampler, then edited directly by evolution from here on
        g["anchor_pairs"] = lb.initial_pairs(g)
    return g


def mutate(g, rng, evolve_heads=False, p=0.5, evolve_pairs=True, p_pair=0.03):
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
    if evolve_pairs:
        # LITERAL ANCHORING. The policy and seed are NOT mutated here: once the genome carries
        # explicit pairs they only ever decided the initial draw, so perturbing them would be a
        # gene with no effect -- a silent no-op that looks like search. Evolution edits the
        # pairs themselves instead. Resize first (NAP/tph may have just moved), then edit.
        h["anchor_pairs"] = resize_pairs(h["anchor_pairs"], lb.n_tables(h),
                                         h["n_anchor_pairs"], rng)
        h["anchor_pairs"], h["_n_pair_edits"] = mutate_pairs(h["anchor_pairs"], rng, p_pair)
    else:
        # policy-only mode (previous behaviour): the categorical IS the anchoring gene
        if rng.random() < 0.15:
            h["anchor_policy"] = str(rng.choice(lb.ANCHOR_POLICIES))
        if rng.random() < 0.2:
            h["anchor_seed"] = int(rng.integers(0, 100000))
    return h


# ----------------------------------------------------------------- fitness
def jsonable(g):
    """A genome with its anchor-pair array turned into nested lists, for checkpointing."""
    d = dict(g)
    if d.get("anchor_pairs") is not None:
        d["anchor_pairs"] = np.asarray(d["anchor_pairs"], np.int64).tolist()
    return d


def fitness(mse, params, lam, tput=0, lam_t=0.0):
    """-MSE - lambda*params - lambda_t*throughput. Selection maximises, as everywhere else.

    RECOMMENDATION, and the default: keep lam_t = 0, i.e. fitness stays MSE + param penalty
    with THROUGHPUT LOGGED RAW. Two reasons.

    First, the two costs are not independent here: throughput = n_heads*tph*n_outputs and
    params = n_heads*tph*2^NAP*n_outputs, so throughput is exactly params / 2^NAP. Penalising
    both means charging tph twice while charging NAP once, which silently biases the search
    toward deep-and-narrow -- the very shape the iso-parameter sweep showed is ~5x WORSE per
    parameter. A throughput term would push against a finding we already have.

    Second, the raw logging makes the penalty unnecessary. Every member records (mse, params,
    throughput), so the front can be re-read on either axis or on both jointly (pareto_2d)
    after the fact. A lambda that is in the fitness cannot be changed after the run; a metric
    that is merely logged can.

    --throughput-penalty is there if you want to overrule that, and it composes additively.
    """
    return -mse - lam * params - lam_t * tput


def pareto(front, cost="params"):
    """Non-dominated (cost, mse) points, cheapest first.

    `cost` selects which axis to read the front on -- "params" or "throughput". Both are logged
    raw for every member, so the front can be re-read on either without re-running anything,
    and neither depends on the lambda that fitness happens to use.
    """
    pts = sorted(front, key=lambda r: (r[cost], r["mse"]))
    out, best = [], float("inf")
    for r in pts:
        if r["mse"] < best - 1e-12:
            out.append(r)
            best = r["mse"]
    return out


def pareto_2d(front, tol=1e-12):
    """Non-dominated on (params, throughput, mse) jointly -- a point survives only if nothing
    else is <= it on all three and < on at least one. This is the front the teacher comparison
    is really about, since params and throughput can move in opposite directions."""
    out = []
    for r in front:
        dominated = False
        for s in front:
            if s is r:
                continue
            if (s["params"] <= r["params"] and s["throughput"] <= r["throughput"]
                    and s["mse"] <= r["mse"] + tol
                    and (s["params"] < r["params"] or s["throughput"] < r["throughput"]
                         or s["mse"] < r["mse"] - tol)):
                dominated = True
                break
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: (r["params"], r["throughput"]))


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
    ap.add_argument("--throughput-penalty", type=float, default=0.0,
                    help="lambda_t in -MSE - lambda*params - lambda_t*throughput. 0 (default, "
                         "and recommended) leaves throughput logged raw -- see fitness()")
    ap.add_argument("--teacher-tol", type=float, default=0.0,
                    help="a member counts as matching the teacher's fit if its held-out MSE is "
                         "within this of the teacher's")
    ap.add_argument("--teacher-steps", type=int, default=None,
                    help="training steps for the teacher reference (default: --steps, so the "
                         "comparison is at equal training budget)")
    ap.add_argument("--no-evolve-anchor-pairs", action="store_true",
                    help="revert to the categorical policy gene instead of evolving the "
                         "literal anchor pairs")
    ap.add_argument("--warm-start", action="store_true",
                    help="LAMARCKIAN: a child inherits its parent's TRAINED row weights, "
                         "remapped into its own shape, instead of cold-starting. Off by "
                         "default, so the cold-start behaviour is unchanged")
    ap.add_argument("--warm-start-std", type=float, default=1e-4,
                    help="std for genuinely NEW cells (tables added by a tph increase). Small "
                         "on purpose -- 10x below the engine's own initial_weights_noise of "
                         "1e-3 -- so new capacity starts neutral and has to earn its weight "
                         "through training rather than arriving with a random head start")
    ap.add_argument("--pair-mutation-rate", type=float, default=0.03,
                    help="per-PAIR probability of an anchor edit, so the edit rate per anchor "
                         "is constant across genome sizes")
    ap.add_argument("--evolve-heads", action="store_true",
                    help="let n_heads vary. Off by default: under a summed readout it is the "
                         "same capacity axis as tables_per_head (see lut_backprop's docstring)")
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

    # THE TEACHER REFERENCE. Trained here, at the same budget as every candidate, so the
    # comparison is like-for-like rather than against a number from a different run.
    ev_pairs = not a.no_evolve_anchor_pairs
    tg = lb.teacher_genome()
    t_params, t_tput = lb.param_count(tg), lb.throughput(tg)
    t_steps = a.teacher_steps or a.steps
    tr = lb.train_eval(tg, Xtr, Ytr, Xte, Yte, t_steps, a.batch, seed=a.seed, device=dev)
    t_mse = tr["heldout_mse"]
    print(f"  TEACHER reference ({lb.genome_str(tg)}):")
    print(f"    held-out MSE {t_mse:.5f}   params {t_params:,}   "
          f"throughput {t_tput:,} weights/forward   ({t_steps} steps)")
    print(f"  anchoring: {'LITERAL PAIRS evolved' if ev_pairs else 'policy gene only'}"
          f"{f', pair mutation rate {a.pair_mutation_rate}' if ev_pairs else ''}")

    M = max(1, int(a.cull * a.pool))
    genomes = [seed_genome(np.random.default_rng(a.seed * 100 + i), a.evolve_heads, ev_pairs)
               for i in range(a.pool)]
    ewma = np.full(a.pool, np.nan)
    age = np.zeros(a.pool, int)
    lineage = np.arange(a.pool)
    # WARM START: the trained row weights of each live pool slot. None until a slot has been
    # trained once, so round 0 is a cold start for everyone either way.
    pool_w = [None] * a.pool
    hist, seen, t0 = [], [], time.time()
    if a.warm_start:
        print(f"  WARM START on: children inherit their parent's trained weights, remapped; "
              f"new cells at std {a.warm_start_std:g}")

    for rnd in range(a.rounds):
        fit = np.zeros(a.pool)
        pre_gain = []
        mses = np.zeros(a.pool)
        pars = np.zeros(a.pool, dtype=np.int64)
        tputs = np.zeros(a.pool, dtype=np.int64)
        for i, g in enumerate(genomes):
            # a FRESH training seed per (round, member): the score is a noisy sample of what
            # the architecture reaches, and the EWMA is what averages it
            r = lb.train_eval(g, Xtr, Ytr, Xte, Yte, a.steps, a.batch,
                              seed=a.seed * 100003 + rnd * 97 + i, device=dev,
                              init_weights=pool_w[i] if a.warm_start else None,
                              return_weights=a.warm_start)
            if a.warm_start:
                # LAMARCKIAN READBACK: the slot now holds this member's TRAINED weights, which
                # is what its children will inherit and what it will itself resume from next
                # round. Without this the pool would keep re-training from the same birth
                # weights and nothing learned would ever accumulate.
                pool_w[i] = r["weights"]
                pre_gain.append(r["pretrain_heldout_mse"])
            mses[i], pars[i], tputs[i] = r["heldout_mse"], r["params"], lb.throughput(g)
            fit[i] = fitness(mses[i], pars[i], a.size_penalty,
                             tputs[i], a.throughput_penalty)
            dom = lb.dominates_teacher(mses[i], pars[i], tputs[i],
                                       t_mse, t_params, t_tput, a.teacher_tol)
            seen.append(dict(rnd=rnd, member=i, mse=float(mses[i]), params=int(pars[i]),
                             throughput=int(tputs[i]), genome=r["genome"], vs_teacher=dom))
            if dom["dominates"]:
                print(f"      >>> member {i} DOMINATES THE TEACHER: MSE {mses[i]:.5f} "
                      f"(teacher {t_mse:.5f}), params {pars[i]:,} (teacher {t_params:,}), "
                      f"throughput {tputs[i]:,} (teacher {t_tput:,})", flush=True)
        ewma = np.where(np.isnan(ewma), fit, (1 - a.alpha) * ewma + a.alpha * fit)
        age += 1

        eligible = np.nonzero(age > a.grace)[0]
        if eligible.size >= M:
            worst = eligible[np.argsort(ewma[eligible])[:M]]
            surv = np.setdiff1d(np.arange(a.pool), worst)
            for slot in worst:
                c1, c2 = rng.choice(surv, 2, replace=False)
                par = c1 if ewma[c1] >= ewma[c2] else c2
                child = mutate(genomes[par], rng, a.evolve_heads,
                               evolve_pairs=ev_pairs, p_pair=a.pair_mutation_rate)
                if a.warm_start and pool_w[par] is not None:
                    # remap the PARENT'S TRAINED weights into the CHILD'S shape. Done here, at
                    # birth, so the child's first training round already starts from them.
                    pool_w[slot] = lb.remap_weights(
                        pool_w[par], genomes[par]["n_anchor_pairs"],
                        child["n_anchor_pairs"], lb.n_tables(child),
                        std=a.warm_start_std, rng=rng)
                elif a.warm_start:
                    pool_w[slot] = None
                genomes[slot] = child
                ewma[slot] = ewma[par]
                age[slot] = 0
                lineage[slot] = lineage[par]

        bi = int(np.nanargmax(ewma))
        pf = pareto(seen)
        # Which anchoring is selection actually keeping? Counted over the LIVE pool after the
        # cull, so it tracks what survived rather than what was tried.
        pol_counts = {p: int(sum(g["anchor_policy"] == p for g in genomes))
                      for p in lb.ANCHOR_POLICIES}
        rec = dict(rnd=rnd, best=float(np.nanmax(ewma)), mean=float(np.nanmean(ewma)),
                   batch_best=float(fit.max()),
                   min_mse=float(mses.min()), min_params=int(pars.min()),
                   median_params=int(np.median(pars)),
                   n_lineages=int(np.unique(lineage).size),
                   anchor_policy_counts=pol_counts,
                   min_throughput=int(tputs.min()),
                   median_throughput=int(np.median(tputs)),
                   throughput_vec=[int(v) for v in tputs],
                   n_dominating_teacher=int(sum(s["vs_teacher"]["dominates"]
                                                for s in seen if s["rnd"] == rnd)),
                   teacher=dict(mse=t_mse, params=t_params, throughput=t_tput),
                   # warm start only: mean held-out MSE BEFORE this round's training. A cold
                   # start sits at the target variance (~1.06); anything far below it is
                   # inherited knowledge that actually survived the remap.
                   pretrain_mse_mean=(float(np.mean(pre_gain)) if pre_gain else None),
                   total_backprop_steps=int((rnd + 1) * a.pool * a.steps),
                   anchor_policy_vec=[g["anchor_policy"] for g in genomes],
                   nap_vec=[g["n_anchor_pairs"] for g in genomes],
                   tph_vec=[g["tables_per_head"] for g in genomes],
                   wall=round(time.time() - t0, 1),
                   fitness_vec=[round(float(v), 6) for v in fit],
                   mse_vec=[round(float(v), 6) for v in mses],
                   params_vec=[int(v) for v in pars],
                   pareto=[dict(params=p["params"], throughput=p["throughput"],
                                mse=round(p["mse"], 6),
                                nap=p["genome"]["n_anchor_pairs"],
                                tph=p["genome"]["tables_per_head"]) for p in pf])
        hist.append(rec)
        print(f"  round {rnd:3d}  best fitness {np.nanmax(ewma):+.5f}  "
              f"min MSE {mses.min():.5f}  params min/med {pars.min():,}/"
              f"{int(np.median(pars)):,}  tput min/med {tputs.min():,}/"
              f"{int(np.median(tputs)):,}  pareto {len(pf)}  "
              f"beats-teacher {rec['n_dominating_teacher']}  {rec['wall']:.0f}s", flush=True)
        json.dump(hist, open(os.path.join(out_dir, f"lut_evolve{a.tag}.json"), "w"), indent=1)
        if a.ckpt:
            json.dump(dict(genomes=[jsonable(g) for g in genomes], ewma=ewma.tolist(),
                           age=age.tolist(), lineage=lineage.tolist(), next_rnd=rnd + 1,
                           seen=seen), open(a.ckpt, "w"))

    print(f"\nBEST member {bi}: fitness {ewma[bi]:+.5f}")
    print(f"  {lb.genome_str(genomes[bi])}")
    def row(p):
        g = p["genome"]
        return (f"  {p['params']:10,} {p['throughput']:11,} {p['mse']:13.5f} "
                f"{p['mse'] / base['constant_predictor_mse']:11.4f}x  "
                f"NAP {g['n_anchor_pairs']:2d} x tph {g['tables_per_head']:3d} "
                f"x heads {g['n_heads']}  lr {g['lr']:.4g}")

    hdr = f"  {'params':>10} {'throughput':>11} {'held-out MSE':>13} {'vs constant':>12}  config"
    for cost, label in (("params", "PARAMETERS"), ("throughput", "THROUGHPUT")):
        pf = pareto(seen, cost)
        print(f"\nPARETO FRONT on {label} ({len(pf)} points, lambda-free):")
        print(hdr)
        for p in pf:
            print(row(p))

    joint = pareto_2d(seen)
    print(f"\nJOINT FRONT, non-dominated on (params, throughput, MSE) together "
          f"({len(joint)} points):")
    print(hdr)
    for p in joint:
        print(row(p))

    # THE HEADLINE CHECK: anything that matches the teacher's fit while costing less on both.
    print(f"\nVS THE TEACHER  (MSE {t_mse:.5f}, params {t_params:,}, "
          f"throughput {t_tput:,}, tol {a.teacher_tol}):")
    dom = [s for s in seen if s["vs_teacher"]["dominates"]]
    both = [s for s in seen if s["vs_teacher"]["both_better"]]
    fit_ok = [s for s in seen if s["vs_teacher"]["fit_ok"]]
    print(f"  {len(fit_ok)}/{len(seen)} candidates matched the teacher's fit; "
          f"{len(dom)} Pareto-dominate it; {len(both)} beat it on BOTH params and throughput.")
    for s in sorted(dom, key=lambda s: (s["params"], s["throughput"]))[:10]:
        g = s["genome"]
        print(f"    MSE {s['mse']:.5f}  params {s['params']:,} "
              f"({s['params'] / t_params:.2f}x)  throughput {s['throughput']:,} "
              f"({s['throughput'] / t_tput:.2f}x)  "
              f"NAP {g['n_anchor_pairs']} x tph {g['tables_per_head']}")
    if not dom:
        best_fit = min(seen, key=lambda s: s["mse"])
        cheap = min((s for s in fit_ok), key=lambda s: s["params"], default=None)
        print(f"    none yet. Best fit overall: MSE {best_fit['mse']:.5f} at "
              f"{best_fit['params']:,} params / {best_fit['throughput']:,} throughput.")
        if cheap:
            print(f"    Cheapest match of the teacher's fit: {cheap['params']:,} params / "
                  f"{cheap['throughput']:,} throughput at MSE {cheap['mse']:.5f}.")

    if not ev_pairs:
        print("\nANCHORING (policy gene mode):")
        for pol in lb.ANCHOR_POLICIES:
            live = sum(g["anchor_policy"] == pol for g in genomes)
            tried = [s["mse"] for s in seen if s["genome"]["anchor_policy"] == pol]
            med = f"median held-out MSE {np.median(tried):.5f}" if tried else "none trained"
            print(f"  {pol:24s} {live:2d}/{len(genomes)} of the final pool   "
                  f"{len(tried):3d} candidates trained, {med}")
    json.dump(dict(baselines=base, history=hist, seen=seen,
                   teacher=dict(mse=t_mse, params=t_params, throughput=t_tput,
                                genome=lb.genome_str(tg), steps=t_steps),
                   pareto_params=pareto(seen, "params"),
                   pareto_throughput=pareto(seen, "throughput"),
                   pareto_joint=joint,
                   evolve_anchor_pairs=ev_pairs,
                   best_genome=jsonable(genomes[bi])),
              open(os.path.join(out_dir, f"lut_evolve{a.tag}_final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
