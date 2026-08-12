"""exp012 growable nets: steady-state evolution where SIZE is part of the genome.

Same loop shape as tiny_evolve (pool of K, all re-scored on one shared batch per round, EWMA
over rounds, cull the worst past a grace period, refill by tournament + crossover + mutate),
with two differences:

  * SELECTION IS ON A COST, not on MSE. fitness = MSE + lam*active_neurons + mu*fanout_excess,
    minimised. The size and fan-out terms are what make growing a decision rather than a free
    lunch: a neuron has to pay for itself.
  * the genome carries an ACTIVE bit per hidden slot, and mutation can grow or shrink.

The pool is seeded from a fixed-size winner: member 0 is that genome placed into the larger
layout UNPERTURBED, and members 1..K-1 are mutations of it. A pool of 512 identical genomes
would have no variation for selection to act on, so the seed enters once and the rest are its
offspring.

    python tiny_grow_evolve.py --rounds 1500 --pool 512 --cull 64 --tag g0 --out-dir ...
"""
import argparse
import json
import os
import time

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder
from tiny_evolve import load_ckpt as load_small_ckpt

SEED_CKPT = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
             "exp012_tiny-direct-genome/full_run_crossover_latinhib/ck_L0.npz")


def small_leader(ck):
    pool, ewma, *_ = load_small_ckpt(ck)
    fin = np.where(np.isfinite(ewma))[0]
    return pool[int(fin[np.argmin(ewma[fin])])]


def save_ckpt(path, pool, ewma, age, rnd, hist, best, rng, era="normalized"):
    # ERA TAG. Checkpoints written before the constant-gain change hold ABSOLUTE weights;
    # these hold NORMALISED ones. Same shapes, same dtypes, silently 200x apart -- so the
    # distinction has to travel with the file rather than with whoever remembers.
    np.savez_compressed(
        path + ".tmp.npz",
        era=np.frombuffer(era.encode(), np.uint8), gain=np.float64(G.GAIN),
        mask=np.stack([g["mask"] for g in pool]).astype(np.uint8),
        delay=np.stack([g["delay"] for g in pool]).astype(np.int16),
        weight=np.stack([g["weight"] for g in pool]).astype(np.float32),
        act_exc=np.stack([g["act_exc"] for g in pool]).astype(np.uint8),
        act_inh=np.stack([g["act_inh"] for g in pool]).astype(np.uint8),
        aff_a=np.stack([G.affine_of(g)[0] for g in pool]),
        aff_b=np.stack([G.affine_of(g)[1] for g in pool]),
        inh_coeff=np.array([G.inh_coeff_of(g) for g in pool]),
        gain_gene=np.array([G.gain_of(g) for g in pool]),
        ewma=ewma, age=age, rnd=np.int64(rnd),
        hist=np.frombuffer(json.dumps(T.jsonable(hist)).encode(), np.uint8),
        best=np.frombuffer(json.dumps(T.jsonable(best)).encode(), np.uint8),
        rng=np.frombuffer(json.dumps(rng.bit_generator.state).encode(), np.uint8))
    os.replace(path + ".tmp.npz", path)


def load_ckpt(path):
    Z = np.load(path, allow_pickle=False)
    pool = [dict(mask=Z["mask"][i].astype(bool), delay=Z["delay"][i].astype(np.int64),
                 weight=Z["weight"][i].astype(np.float64),
                 act_exc=Z["act_exc"][i].astype(bool), act_inh=Z["act_inh"][i].astype(bool),
                 aff_a=(Z["aff_a"][i] if "aff_a" in Z.files else np.full(G.N_TARGET, G.AFF_A_INIT)),
                 aff_b=(Z["aff_b"][i] if "aff_b" in Z.files
                        else np.full(G.N_TARGET, G.AFF_B_INIT)),
                 inh_coeff=(float(Z["inh_coeff"][i]) if "inh_coeff" in Z.files
                            else G.INH_COEFF_LEGACY),
                 gain=(float(Z["gain_gene"][i]) if "gain_gene" in Z.files else float(G.GAIN)))
            for i in range(len(Z["mask"]))]
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(Z["rng"].tobytes().decode())
    era = Z["era"].tobytes().decode() if "era" in Z.files else "absolute"
    if era == "absolute":
        # an untagged checkpoint predates the constant-gain change, so its weights are
        # absolute; convert rather than silently running them 200x too hot
        pool = [G.normalize_abs(g) for g in pool]
    return (pool, Z["ewma"], Z["age"], int(Z["rnd"]),
            json.loads(Z["hist"].tobytes().decode()),
            json.loads(Z["best"].tobytes().decode()), rng)


def genome_to_json(g):
    return dict(mask=g["mask"].astype(np.int8).tolist(),
                delay=g["delay"].astype(int).tolist(),
                weight=np.round(g["weight"], 6).tolist(),
                aff_a=np.round(G.affine_of(g)[0], 6).tolist(),
                aff_b=np.round(G.affine_of(g)[1], 6).tolist(),
                inh_coeff=round(G.inh_coeff_of(g), 6),
                gain=round(G.gain_of(g), 6),
                act_exc=g["act_exc"].astype(int).tolist(),
                act_inh=g["act_inh"].astype(int).tolist())


def tournament(rng, fit, k=2):
    i = rng.integers(0, len(fit), k)
    return int(i[np.argmax(fit[i])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--cull", type=int, default=64)
    ap.add_argument("--grace", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ewma", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=G.W_SIGMA_NORM,
                    help="normalised weight-step sigma; 0.045 == the old 9.0 absolute")
    ap.add_argument("--gain", type=float, default=G.GAIN)
    ap.add_argument("--lam", type=float, default=0.35)
    ap.add_argument("--mu", type=float, default=0.10)
    ap.add_argument("--cap-floor", type=int, default=6)
    ap.add_argument("--cap-frac", type=float, default=0.10)
    ap.add_argument("--p-add", type=float, default=0.02)
    ap.add_argument("--p-prune", type=float, default=0.02)
    ap.add_argument("--p-delay", type=float, default=0.08)
    ap.add_argument("--p-weight", type=float, default=None,
                    help="default 0.25, but 0.5 when a 2-level-per-Dale-half weight grid is "
                         "active, because a clamped hop on a binary sub-grid is a no-op "
                         "half the time (measured 0.485 moves per proposal)")
    ap.add_argument("--p-grow", type=float, default=0.03)
    ap.add_argument("--p-shrink", type=float, default=0.03)
    ap.add_argument("--p-inhcoeff", type=float, default=0.0,
                    help="per-mutate probability of stepping the global inhibition "
                         "coefficient. 0 = the gene never moves (default, backward compatible)")
    ap.add_argument("--p-gain", type=float, default=0.0,
                    help="per-mutate probability of stepping the global synaptic gain. "
                         "0 = frozen at 200.0 (default, backward compatible)")
    ap.add_argument("--gain-evolve", action="store_true",
                    help="shorthand for --p-gain 0.25")
    ap.add_argument("--inhibition-coeff-evolve", action="store_true",
                    help="shorthand for --p-inhcoeff 0.25")
    ap.add_argument("--lif-tau", type=float, default=None,
                    help="use LIF neurons with this membrane time constant (threshold 1.0). "
                         "Omit for the Izhikevich metas, which stay the default.")
    ap.add_argument("--t-in", type=int, default=None,
                    help="input tick window (default 32). Widening it spreads the latency "
                         "code so spike-order circuits get a usable gap between inputs.")
    ap.add_argument("--n-ticks", type=int, default=None, help="episode length (default 96)")
    ap.add_argument("--readout-window", type=int, default=None, help="default 32")
    ap.add_argument("--max-delay", type=int, default=None,
                    help="widen the synapse delay bank past 64")
    ap.add_argument("--readout", default="evolved",
                    choices=("evolved", "linear", "diagls", "lut", "wta"),
                    help="'evolved' = the 12 diagonal affine genes (historical). 'linear' = "
                         "a full N_OUT x N_OUT least-squares map fitted on each round's "
                         "TRAINING batch and carried to held-out.")
    ap.add_argument("--p-affine", type=float, default=0.25,
                    help="per-output-dimension rate for the readout-calibration step; "
                         "0 disables the affine genes entirely")
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed-ckpt", default=SEED_CKPT)
    ap.add_argument("--max-episode-batch", type=int, default=G.MAX_EPISODE_BATCH,
                    help="episode chunk size. The engine caps batch x neurons x ticks, and "
                         "a concurrently-running job eats into the headroom, so lower this "
                         "when sharing the GPU. Chunking is exact -- every metric is a mean "
                         "over samples.")
    ap.add_argument("--fanout-cap", type=int, default=0,
                    help="hard ceiling on outgoing synapses per HIDDEN neuron, enforced by "
                         "refusal in add/grow. 0 = off (default), so continuous runs are "
                         "unchanged. Inputs and outputs are unaffected.")
    ap.add_argument("--weight-levels", default=None,
                    help='comma-separated weight grid, e.g. "-0.5,0,1.0". Defaults to the '
                         '11-level +-1.0 set. Dale-split at startup.')
    ap.add_argument("--delay-levels", default=None,
                    help='delay grid: "odd" for {1,3,..,63}, or a comma-separated list. '
                         'Omit for free integers in [1,64] (the historical behaviour).')
    ap.add_argument("--out-per-target", type=int, default=1,
                    help="K output NEURONS per target dimension. Each target reads a fixed "
                         "aggregate of its OWN group only -- strictly non-mixing.")
    ap.add_argument("--out-agg", default="mean", choices=("mean", "min"))
    ap.add_argument("--bit-task", default=None, metavar="A,B",
                    help="score against the single sign-comparison bit "
                         "1[x_norm[A] > x_norm[B]] instead of the action. Under the latency "
                         "encoder this is 'which of two input spikes arrives first'. "
                         "Off by default.")
    ap.add_argument("--lut-target", type=int, default=None, metavar="DIM",
                    help="reshape the target into a 32-entry LUT indexed by the output's "
                         "first-spike time: equal-population bins on DIM, LUT[b] = that "
                         "bin's mean. Fitted on the TRAINING pool only. Use with "
                         "--readout lut, which decodes through the same LUT and fits "
                         "nothing. Off by default; the offset target is unchanged.")
    ap.add_argument("--hidden-capacity", default=None,
                    help="EXC,INH -- resize the hidden layer itself (default 40,10). Unlike "
                         "--init-exc/--init-inh this shrinks what build() allocates, so the "
                         "reported topology is the built one. INH 0 removes inhibition "
                         "entirely; do not pass --inhibition-coeff-evolve with it.")
    ap.add_argument("--target-dims", default=None,
                    help="comma-separated subset of the six target dimensions to be scored "
                         "on, e.g. '5' for a single-output task. Omit for all six. NOTE the "
                         "baseline moves with it: the run is judged against the constant "
                         "predictor on THESE dimensions, not the 6-dim 34.15.")
    ap.add_argument("--quantized", action="store_true",
                    help="snap every weight to the 11-level +-1.0 grid (0.2 spacing, Dale-"
                         "split) and mutate by discrete level hops instead of a Gaussian")
    ap.add_argument("--random-init", action="store_true",
                    help="cold start: 512 independent random_genome() draws instead of a "
                         "seeded pool. No prior checkpoint enters the lineage.")
    ap.add_argument("--init-exc", type=int, default=8)
    ap.add_argument("--init-inh", type=int, default=2)
    ap.add_argument("--p-init", type=float, default=0.5,
                    help="edge density of a random genome; 0.5 over the active-legal cells "
                         "gives ~140 synapses at 8+2 active, matching the seed's 134")
    ap.add_argument("--seed-all-slots", action="store_true",
                    help="activate ALL 40+10 hidden slots at seeding and sparse-wire the "
                         "newcomers; with --p-grow 0 --p-shrink 0 this gives a FIXED "
                         "full-capacity run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="g0")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    # BEFORE any genome exists. set_target_dims resizes the output layer itself, so when both
    # are given it has to run second -- it reads OUT_PER_TARGET back out of the module.
    # set_hidden_capacity is first of all: the other two rebuild geometry that depends on it.
    if a.t_in or a.n_ticks or a.max_delay:
        G.set_episode(t_in=a.t_in, n_ticks=a.n_ticks,
                      readout_window=a.readout_window, d_hi=a.max_delay)
    if a.lif_tau:
        G.set_lif(tau=a.lif_tau, threshold=1.0, v_rest=0.0, v_reset=0.0, refractory_ticks=0)
        # LIF fires at v >= 1.0, where Izhikevich needed ~200 of current, so the gain gene
        # has to live around 1 instead of 200 or every genome saturates on its first spike.
        G.GAIN = G.GAIN_INIT = 1.0
        G.GAIN_LO, G.GAIN_HI, G.GAIN_SIGMA = 0.2, 10.0, 0.15
    if a.hidden_capacity:
        ne, ni = (int(x) for x in a.hidden_capacity.split(","))
        assert not (ni == 0 and a.inhibition_coeff_evolve), \
            "inhibition_coeff is meaningless with 0 inhibitory neurons"
        G.set_hidden_capacity(ne, ni)
    if a.out_per_target != 1:
        G.set_out_per_target(a.out_per_target, a.out_agg)
    if a.target_dims:
        G.set_target_dims([int(x) for x in a.target_dims.split(",")])
    if a.max_delay:
        # DLY_HI is per TARGET COLUMN, and set_out_per_target / set_target_dims rebuild it,
        # so the widened range has to be applied AFTER them rather than inside set_episode.
        # Only the excitatory columns widen: input->exc must be able to span the whole
        # 128-tick input window, while hidden->out stays in [1, 64] so a hidden spike still
        # lands inside the readout window instead of past the end of the episode.
        G.DLY_HI[:] = G.D_HI
        G.DLY_HI[G.C_EXC] = int(a.max_delay)
        print(f"delay ranges: exc columns [1,{int(a.max_delay)}], others [1,{G.D_HI}]",
              flush=True)
    G.MAX_EPISODE_BATCH = a.max_episode_batch
    if a.weight_levels:
        G.set_weight_levels([float(x) for x in a.weight_levels.split(",")])
    if a.delay_levels:
        G.set_delay_levels(list(range(1, 64, 2)) if a.delay_levels == "odd"
                           else [int(x) for x in a.delay_levels.split(",")])
    if a.inhibition_coeff_evolve and not a.p_inhcoeff:
        a.p_inhcoeff = 0.25
    if a.gain_evolve and not a.p_gain:
        a.p_gain = 0.25
    if a.p_weight is None:
        # resolved AFTER the grid is installed, so it reflects the grid actually in use
        binary = min(len(G.QUANT_POS), len(G.QUANT_NEG)) <= 2
        a.p_weight = 0.5 if (a.quantized and binary) else 0.25
    G.QUANTIZED = a.quantized          # set BEFORE any genome is created
    G.FANOUT_CAP = a.fanout_cap if a.fanout_cap > 0 else None
    era = "quantized" if a.quantized else "normalized"

    os.makedirs(a.out_dir, exist_ok=True)
    ck = os.path.join(a.out_dir, f"ck_{a.tag}.npz")
    beat = os.path.join(a.out_dir, f"{a.tag}.progress")

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    if a.lut_target is not None:
        # Bins and decode values come from the TRAINING pool only and are then frozen; the
        # held-out split is binned with those edges and never contributes to either.
        from tiny_lut_target import build_lut
        edges, lut, _ = build_lut(Yp[:, a.lut_target])
        tbl = np.concatenate([lut, [float(lut[np.digitize(Yp[:, a.lut_target],
                                                          edges)].mean())]])
        G.set_lut_task(edges, tbl, a.lut_target)
        print(f"LUT target on dim {a.lut_target}: 32 equal-population bins, "
              f"silence decodes to the training mean {tbl[-1]:.4f}", flush=True)
    if a.bit_task:
        ia, ib = (int(v) for v in a.bit_task.split(","))
        G.set_bit_task(ia, ib)
        print(f"BIT task: target = 1[ x_norm[{ia}] > x_norm[{ib}] ]  "
              f"P(1) train {float((Xp[:, ia] > Xp[:, ib]).mean()):.4f}  "
              f"held-out {float((Xv[:, ia] > Xv[:, ib]).mean()):.4f}", flush=True)
    base_v = G.task_baseline(Yv, Xv)      # == T.constant_baseline when all six dims are on
    tgt_v = G.task_targets(Yv, Xv)

    def cost(g, mse):
        return G.cost_terms(g, mse, a.lam, a.mu, a.cap_floor, a.cap_frac)

    start = 0
    if a.resume and os.path.exists(ck):
        pool, ewma, age, start, hist, best, rng = load_ckpt(ck)
        start += 1
        print(f"resumed {ck} at round {start}", flush=True)
    elif a.random_init:
        # COLD START: 512 independent random genomes, no checkpoint anywhere in the lineage.
        # random_genome already respects every invariant (Dale by source row, LEGAL, the
        # delay-1 pin, the active mask) and starts affine genes at the identity, so a random
        # pool is structurally the same kind of object as a seeded one -- just untrained.
        pool = [G.random_genome(rng, p_init=a.p_init, n_exc=a.init_exc, n_inh=a.init_inh)
                for _ in range(a.pool)]
        n0 = int(np.mean([g["mask"].sum() for g in pool]))
        ewma = np.full(a.pool, np.nan)
        age = np.zeros(a.pool, np.int64)
        hist, best = [], dict(fitness=np.inf)
        print(f"RANDOM init: {a.pool} independent genomes, {a.init_exc} exc + {a.init_inh} "
              f"inh active (E-frac {a.init_exc / (a.init_exc + a.init_inh):.3f}), "
              f"~{n0} synapses each", flush=True)
    else:
        seed_g = G.seed_from_small(small_leader(a.seed_ckpt))
        if a.seed_all_slots:
            seed_g = G.activate_all(seed_g, rng)
        pool = [seed_g] + [G.mutate(seed_g, rng, a.p_add, a.p_prune, a.p_delay,
                                    a.p_weight, a.sigma, a.p_grow, a.p_shrink, a.p_affine,
                                    a.p_inhcoeff, a.p_gain)
                           for _ in range(a.pool - 1)]
        ewma = np.full(a.pool, np.nan)
        age = np.zeros(a.pool, np.int64)
        hist, best = [], dict(fitness=np.inf)
        print(f"seeded from {a.seed_ckpt}: {int(seed_g['mask'].sum())} syn, "
              f"{G.n_active(seed_g)} active, E-frac {G.e_fraction(seed_g):.3f}", flush=True)

    n_sex = 0
    for rnd in range(start, a.rounds):
        Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, rnd)
        H = G.build(pool, device=a.device, seed=1)
        s = G.score(H, Xb, Yb, enc, genomes=pool, readout=a.readout)
        mse = s["mse"]
        terms = [cost(g, m) for g, m in zip(pool, mse)]
        fit = np.array([t["fitness"] for t in terms])
        ewma = np.where(np.isnan(ewma), fit, a.ewma * fit + (1 - a.ewma) * ewma)
        age += 1

        i = int(np.argmin(fit))
        if fit[i] < best["fitness"]:
            Hb = G.build([pool[i]], device=a.device)
            if a.readout in ("linear", "diagls"):
                st = G.score(Hb, Xb, Yb, enc, genomes=[pool[i]], readout=a.readout)
                sv = G.score(Hb, Xv, Yv, enc, genomes=[pool[i]], readout=a.readout,
                             readout_map=st["readout_map"])
            else:
                sv = G.score(Hb, Xv, Yv, enc, genomes=[pool[i]])
            r_b, ceil_b = T.affine_ceiling_and_r(sv["first"][:, 0, :], tgt_v)
            best = dict(rnd=rnd, batch_fitness=float(fit[i]), batch_mse=float(mse[i]),
                        heldout_mse=float(sv["mse"][0]), heldout_tau=float(sv["tau"][0]),
                        heldout_silent=float(sv["silent"][0]),
                        heldout_mean_abs_r=r_b, heldout_affine_ceiling=ceil_b,
                        terms=terms[i], stats=G.genome_stats(pool[i]),
                        fitness=float(fit[i]), genome=genome_to_json(pool[i]))

        na = np.array([G.n_active(g) for g in pool])
        ns = np.array([int(g["mask"].sum()) for g in pool])
        rec = dict(rnd=rnd, fit_min=float(fit.min()), fit_mean=float(fit.mean()),
                   mse_min=float(mse.min()), mse_of_best=float(mse[i]),
                   n_active_mean=float(na.mean()), n_active_min=int(na.min()),
                   n_active_max=int(na.max()), n_active_of_best=int(na[i]),
                   n_exc_mean=float(np.mean([g["act_exc"].sum() for g in pool])),
                   n_inh_mean=float(np.mean([g["act_inh"].sum() for g in pool])),
                   e_frac_mean=float(np.mean([G.e_fraction(g) for g in pool])),
                   n_syn_mean=float(ns.mean()), n_syn_of_best=int(ns[i]),
                   fanout_excess_of_best=terms[i]["fanout_excess"],
                   tau_of_best=float(s["tau"][i]), silent=float(s["silent"].mean()),
                   best_heldout=best.get("heldout_mse"))
        if a.eval_every and rnd % a.eval_every == 0:
            j = int(np.argmin(ewma))
            Hb = G.build([pool[j]], device=a.device)
            if a.readout in ("linear", "diagls"):
                st = G.score(Hb, Xb, Yb, enc, genomes=[pool[j]], readout=a.readout)
                sv = G.score(Hb, Xv, Yv, enc, genomes=[pool[j]], readout=a.readout,
                             readout_map=st["readout_map"])
            else:
                sv = G.score(Hb, Xv, Yv, enc, genomes=[pool[j]])
            r_l, ceil_l = T.affine_ceiling_and_r(sv["first"][:, 0, :], tgt_v)
            rec.update(leader_heldout_mse=float(sv["mse"][0]),
                       leader_heldout_tau=float(sv["tau"][0]),
                       leader_mean_abs_r=r_l, leader_affine_ceiling=ceil_l,
                       leader_n_active=G.n_active(pool[j]),
                       leader_n_syn=int(pool[j]["mask"].sum()))
        hist.append(rec)

        elig = np.where(age > a.grace)[0]
        if len(elig) > a.cull:
            worst = elig[np.argsort(ewma[elig])[::-1][:a.cull]]     # highest cost = worst
            for j in worst:
                p = tournament(rng, -ewma)
                if a.crossover:
                    q = tournament(rng, -ewma)
                    kid = pool[p] if q == p else G.crossover(pool[p], pool[q], rng)
                    n_sex += (q != p)
                else:
                    kid = pool[p]
                pool[j] = G.mutate(kid, rng, a.p_add, a.p_prune, a.p_delay,
                                   a.p_weight, a.sigma, a.p_grow, a.p_shrink, a.p_affine,
                                   a.p_inhcoeff, a.p_gain)
                ewma[j], age[j] = np.nan, 0

        with open(beat + ".tmp", "w") as f:
            f.write(f"{rnd + 1} {a.rounds} {best.get('heldout_mse', float('nan')):.4f}\n")
        os.replace(beat + ".tmp", beat)
        if a.ckpt_every and (rnd % a.ckpt_every == 0 or rnd == a.rounds - 1):
            save_ckpt(ck, pool, ewma, age, rnd, hist, best, rng, era)
        if rnd % 25 == 0 or rnd == a.rounds - 1:
            print(f"r{rnd:4d}  fit {fit.min():7.3f}  mse {mse.min():7.3f}  "
                  f"neurons {na.mean():5.2f} (min {na.min()} max {na.max()})  "
                  f"syn {ns.mean():6.1f}  held-out best "
                  f"{best.get('heldout_mse', float('nan')):7.3f}  "
                  f"[{time.time() - t0:6.1f}s]", flush=True)

    out = dict(config=vars(a), constant_baseline_val=base_v, best=best,
               n_recombined_offspring=int(n_sex), elapsed_s=time.time() - t0)
    with open(os.path.join(a.out_dir, f"{a.tag}_final.json"), "w") as f:
        json.dump(T.jsonable(out), f, indent=1)
    with open(os.path.join(a.out_dir, f"{a.tag}.json"), "w") as f:
        json.dump(T.jsonable(hist), f)
    print(f"DONE: held-out MSE {best['heldout_mse']:.3f} vs constant {base_v:.3f} "
          f"({out['elapsed_s'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
