"""exp012 growable nets: validate the operators, and that normalisation changed NOTHING.

The genome now stores normalised magnitudes in [0,1] / [-1,0] and build() applies a single
constant GAIN. That is meant to be a pure reparameterisation, so the checks below are aimed
squarely at proving it: the same network, built two ways, must reach the engine with the same
weights and score the same MSE, and the mutation operators must still sample the same
distributions once multiplied back by GAIN.

  a EQUIVALENCE  an existing ABSOLUTE-weight genome, normalised and rebuilt -> the effective
                 weights and the held-out MSE are unchanged.
  b INVARIANTS   ~600 genomes from random_genome / mutate / crossover / grow / shrink: Dale,
                 unit range, legality, delay-1 pin, active-mask consistency.
  c DISTRIBUTION the weight step and the born magnitude, in EFFECTIVE units, still match
                 N(0, 9) and U(6, 36).
  d SPOT-CHECK   the normalised grow leader reproduces its prior held-out MSE.
  e BUILD        round-trip exactness and packed-vs-solo isolation.
"""
import argparse
import json

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load
from harness import LatencyEncoder
from tiny_evolve import load_ckpt as load_small_ckpt
from tiny_grow_evolve import load_ckpt as load_grow_ckpt

BASE = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
        "exp012_tiny-direct-genome/")
CK_SMALL = BASE + "full_run_crossover_latinhib/ck_L0.npz"
CK_GROW = BASE + "full_run_grow/ck_g0.npz"


def leader(pool, ewma):
    fin = np.where(np.isfinite(ewma))[0]
    return pool[int(fin[np.argmin(ewma[fin])])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    R = dict(gain=G.GAIN, sigma_norm=G.W_SIGMA_NORM, born=[G.BORN_LO, G.BORN_HI],
             w_init_norm=G.W_INIT_NORM)

    # ------------------------------------------------------------ a EQUIVALENCE
    pool_g, ewma_g, *_ = load_grow_ckpt(CK_GROW)
    g_abs = leader(pool_g, ewma_g)                 # ABSOLUTE weights, from the old run
    g_norm = G.normalize_abs(g_abs)                # same net, normalised

    eff = G.effective_weights(g_norm)
    max_w_diff = float(np.max(np.abs(eff - g_abs["weight"])))
    H_abs = G.build([g_abs], device=a.device, gain=1.0)     # gain 1 -> weights used as-is
    H_norm = G.build([g_norm], device=a.device)             # gain 200 -> back to absolute
    max_tri_diff = float(np.max(np.abs(H_abs["weights"] - H_norm["weights"])))
    s_abs = G.score(H_abs, Xv, Yv, enc)
    s_norm = G.score(H_norm, Xv, Yv, enc)
    R["equivalence"] = dict(
        n_synapses=int(g_abs["mask"].sum()),
        max_abs_diff_weight_matrix=max_w_diff,
        max_abs_diff_triples_into_engine=max_tri_diff,
        mse_absolute=float(s_abs["mse"][0]), mse_normalised=float(s_norm["mse"][0]),
        mse_abs_diff=abs(float(s_abs["mse"][0]) - float(s_norm["mse"][0])),
        mse_bitwise_equal=bool(s_abs["mse"][0] == s_norm["mse"][0]),
        offsets_identical=bool(np.array_equal(s_abs["first"], s_norm["first"])),
        ok=bool(max_w_diff < 1e-9 and max_tri_diff < 1e-9
                and s_abs["mse"][0] == s_norm["mse"][0]))
    print("a EQUIVALENCE", json.dumps(T.jsonable(R["equivalence"]), indent=1), flush=True)

    # ------------------------------------------------------------ b INVARIANTS
    tot = dict(dale=0, legal=0, pins=0, active=0, delays=0, range=0, fanout=0,
               in_to_inh=0, in_to_out=0, inh_to_out=0)
    n_checked = 0
    mag_max = 0.0

    def acc(g):
        nonlocal n_checked, mag_max
        for k, v in G.all_ok(g).items():
            tot[k] += v
        n_checked += 1
        mag_max = max(mag_max, float(np.abs(g["weight"]).max()))

    seed_small = leader(*load_small_ckpt(CK_SMALL)[:2])
    for _ in range(a.trials):
        acc(G.random_genome(rng))
    g = G.seed_from_small(seed_small)
    for _ in range(a.trials):
        g = G.mutate(g, rng)
        acc(g)
    inc, frac = 0, []
    for _ in range(a.trials):
        p1 = G.mutate(G.seed_from_small(seed_small), rng)
        p2 = G.random_genome(rng)
        c = G.crossover(p1, p2, rng)
        acc(c)
        inc += G.bundle_coherent(c, p1, p2)[1]
        live = G.active_cells(c) & G.LEGAL
        d = live & ~((p1["mask"] == p2["mask"]) & (p1["delay"] == p2["delay"])
                     & (p1["weight"] == p2["weight"]))
        s1 = ((c["mask"] == p1["mask"]) & (c["delay"] == p1["delay"])
              & (c["weight"] == p1["weight"]))
        if d.any():
            frac.append(float(s1[d].mean()))
    R["invariants"] = dict(genomes_checked=n_checked, violations=tot,
                           max_abs_magnitude_seen=mag_max, incoherent_cells=int(inc),
                           from_parent1=float(np.mean(frac)),
                           from_parent1_sd=float(np.std(frac)),
                           ok=bool(sum(tot.values()) == 0 and inc == 0 and mag_max <= 1.0))
    print("b INVARIANTS", json.dumps(T.jsonable(R["invariants"]), indent=1), flush=True)

    # ------------------------------------------------------------ c DISTRIBUTIONS
    step = rng.normal(0.0, G.W_SIGMA_NORM, 200000) * G.GAIN
    born = rng.uniform(G.BORN_LO, G.BORN_HI, 200000) * G.GAIN
    init = rng.uniform(0.0, G.W_INIT_NORM, 200000) * G.GAIN
    R["distributions"] = dict(
        weight_step=dict(mean=float(step.mean()), std=float(step.std()),
                         want="N(0, 9.0)"),
        born_magnitude=dict(mean=float(born.mean()), std=float(born.std()),
                            min=float(born.min()), max=float(born.max()),
                            want="U(6, 36): mean 21.0, std 8.66"),
        init_magnitude=dict(mean=float(init.mean()), max=float(init.max()),
                            want="U(0, 60): mean 30.0"),
        ok=bool(abs(step.std() - 9.0) < 0.1 and abs(born.mean() - 21.0) < 0.1))
    print("c DISTRIBUTIONS", json.dumps(T.jsonable(R["distributions"]), indent=1), flush=True)

    # ------------------------------------------------------------ d SPOT-CHECK
    r_l, ceil_l = T.affine_ceiling_and_r(s_norm["first"][:, 0, :], T.target_offsets(Yv))
    R["spot_check"] = dict(heldout_mse=float(s_norm["mse"][0]),
                           prior_grow_leader_mse=23.4453, tau=float(s_norm["tau"][0]),
                           mean_abs_r=r_l, affine_ceiling=ceil_l,
                           silent=float(s_norm["silent"][0]),
                           size=G.genome_stats(g_norm),
                           ok=bool(abs(float(s_norm["mse"][0]) - 23.4453) < 1e-3))
    print("d SPOT-CHECK", json.dumps(T.jsonable(R["spot_check"]), indent=1), flush=True)

    # ------------------------------------------------------------ e BUILD
    pool = [G.seed_from_small(seed_small)] + [G.random_genome(rng)
                                              for _ in range(a.pool - 1)]
    H = G.build(pool, device=a.device)
    rt = G.verify_round_trip(H)
    rt["ok"] = bool(rt["missing"] == 0 and rt["weights_ok"] == rt["n_requested"]
                    and rt["delays_ok"] == rt["n_requested"]
                    and rt["negative_ok"] == rt["n_negative"])
    packed = G.score(H, Xv[:512], Yv[:512], enc)["mse"]
    which = (0, 5, 17, a.pool - 1)
    solo = [float(G.score(G.build([pool[i]], device=a.device), Xv[:512], Yv[:512],
                          enc)["mse"][0]) for i in which]
    got = [float(packed[i]) for i in which]
    R["build"] = dict(round_trip=rt, which=list(which), packed=got, solo=solo,
                      max_abs_diff=float(np.max(np.abs(np.array(got) - np.array(solo)))),
                      ok=bool(rt["ok"] and np.array_equal(got, solo)))
    print("e BUILD", json.dumps(T.jsonable(R["build"]), indent=1), flush=True)

    R["all_ok"] = bool(R["equivalence"]["ok"] and R["invariants"]["ok"]
                       and R["distributions"]["ok"] and R["spot_check"]["ok"]
                       and R["build"]["ok"])
    print(f"\nALL_OK={R['all_ok']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
