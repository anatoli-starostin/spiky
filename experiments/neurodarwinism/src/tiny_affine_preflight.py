"""exp012: validate the evolvable readout calibration before it is used in anger.

  a IDENTITY    existing leaders, with the default a=1 b=0, reproduce their MSE bit for bit.
  b INVARIANTS  ~400 genomes: the structural invariants are untouched and the affine params
                stay finite and inside their box.
  c ADJUSTS FAST  the key test. Freeze a net -- wiring, weights, delays, all of it -- and
                evolve ONLY the 12 affine genes. How many rounds to reach the analytic
                optimum, and what does the curve look like?
  d ANALYTIC    the closed-form least-squares (a, b), fitted BOTH ways: on held-out (which
                reproduces the reported affine ceiling and is an optimistic bound) and on a
                training batch (the honest target an evolved calibration can actually reach).
"""
import argparse
import json

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder
from tiny_evolve import load_ckpt as load_small_ckpt
from tiny_grow_evolve import load_ckpt as load_grow_ckpt

BASE = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
        "exp012_tiny-direct-genome/")


def leader(pool, ewma):
    fin = np.where(np.isfinite(ewma))[0]
    return pool[int(fin[np.argmin(ewma[fin])])]


def mse_of(y, tgt):
    return float(((y - tgt) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    tgt_v = T.target_offsets(Yv)
    R = dict(sigma_a=G.SIGMA_A, sigma_b=G.SIGMA_B,
             a_limit=G.AFF_A_LIM, b_limit=G.AFF_B_LIM)

    # ---------------------------------------------------------------- a IDENTITY
    g_norm = leader(*load_grow_ckpt(BASE + "full_run_grow_norm/ck_n0.npz")[:2])
    g_lat = G.seed_from_small(leader(
        *load_small_ckpt(BASE + "full_run_crossover_latinhib/ck_L0.npz")[:2]))
    ident = []
    for nm, g, want in (("grow-norm leader", g_norm, 23.71066666666667),
                        ("lat-inhib leader", g_lat, 24.271333333333335)):
        H = G.build([g], device=a.device)
        m_no = float(G.score(H, Xv, Yv, enc)["mse"][0])                    # no genomes given
        m_id = float(G.score(H, Xv, Yv, enc, genomes=[G.with_identity_affine(g)])["mse"][0])
        ident.append(dict(net=nm, mse_no_affine_arg=m_no, mse_identity_affine=m_id,
                          prior_reported=want, diff_vs_prior=abs(m_id - want),
                          bitwise_equal=bool(m_no == m_id)))
    R["identity"] = dict(rows=ident,
                         max_mse_diff=max(x["diff_vs_prior"] for x in ident),
                         ok=bool(all(x["bitwise_equal"] and x["diff_vs_prior"] < 1e-9
                                     for x in ident)))
    print("a IDENTITY", json.dumps(T.jsonable(R["identity"]), indent=1), flush=True)

    # ---------------------------------------------------------------- b INVARIANTS
    tot = dict(dale=0, legal=0, pins=0, active=0, delays=0, range=0, fanout=0,
               in_to_inh=0, in_to_out=0, inh_to_out=0)
    aff_bad = 0
    amin = amax = bmin = bmax = 0.0
    n = 0

    def acc(g):
        nonlocal aff_bad, amin, amax, bmin, bmax, n
        for k, v in G.all_ok(g).items():
            tot[k] += v
        aa, bb = G.affine_of(g)
        aff_bad += int((~np.isfinite(aa)).sum() + (~np.isfinite(bb)).sum()
                       + (np.abs(aa) > G.AFF_A_LIM).sum() + (np.abs(bb) > G.AFF_B_LIM).sum())
        amin, amax = min(amin, float(aa.min())), max(amax, float(aa.max()))
        bmin, bmax = min(bmin, float(bb.min())), max(bmax, float(bb.max()))
        n += 1

    g = G.with_identity_affine(g_norm)
    for _ in range(a.trials):
        g = G.mutate(g, rng)
        acc(g)
    for _ in range(a.trials // 2):
        c = G.crossover(G.mutate(g_norm, rng), G.random_genome(rng), rng)
        acc(c)
    R["invariants"] = dict(genomes_checked=n, violations=tot,
                           affine_out_of_range_or_nonfinite=aff_bad,
                           a_range_seen=[amin, amax], b_range_seen=[bmin, bmax],
                           ok=bool(sum(tot.values()) == 0 and aff_bad == 0))
    print("b INVARIANTS", json.dumps(T.jsonable(R["invariants"]), indent=1), flush=True)

    # ---------------------------------------------------------------- d ANALYTIC (first)
    H = G.build([g_norm], device=a.device)
    raw_v = G.score(H, Xv, Yv, enc)["first"][:, 0, :]
    Xb, Yb, _ = sample_batch(Xp, Yp, 2000, a.seed, 7)
    raw_t = G.score(G.build([g_norm], device=a.device), Xb, Yb, enc)["first"][:, 0, :]
    tgt_t = T.target_offsets(Yb)

    a_val, b_val = G.analytic_affine(raw_v, tgt_v)          # fitted ON held-out: optimistic
    a_tr, b_tr = G.analytic_affine(raw_t, tgt_t)            # fitted on training: honest
    mse_raw = mse_of(raw_v, tgt_v)
    mse_val_fit = mse_of(a_val * raw_v + b_val, tgt_v)
    mse_tr_fit = mse_of(a_tr * raw_v + b_tr, tgt_v)
    R["analytic"] = dict(
        mse_uncalibrated=mse_raw,
        mse_affine_fitted_on_heldout=mse_val_fit, reported_affine_ceiling=16.2216,
        mse_affine_fitted_on_training=mse_tr_fit,
        leakage_gap=mse_tr_fit - mse_val_fit,
        a_train=a_tr.tolist(), b_train=b_tr.tolist())
    print("d ANALYTIC", json.dumps(T.jsonable(R["analytic"]), indent=1), flush=True)

    # ---------------------------------------------------------------- c ADJUSTS FAST
    # the network is frozen; only the 12 affine genes vary. Selection sees TRAINING batches
    # only, and held-out is recorded for reporting -- same discipline as the real runs.
    target = mse_tr_fit
    pool = [G.with_identity_affine(g_norm) for _ in range(a.pool)]
    for k in range(1, a.pool):
        pool[k] = {**pool[k],
                   "aff_a": np.clip(pool[k]["aff_a"] + rng.normal(0, G.SIGMA_A, G.N_OUT),
                                    -G.AFF_A_LIM, G.AFF_A_LIM),
                   "aff_b": np.clip(pool[k]["aff_b"] + rng.normal(0, G.SIGMA_B, G.N_OUT),
                                    -G.AFF_B_LIM, G.AFF_B_LIM)}
    Hfix = G.build([g_norm], device=a.device)               # ONE frozen net, reused
    raw_cache = {}

    def raw_for(rnd):
        if rnd not in raw_cache:
            Xr, Yr, _ = sample_batch(Xp, Yp, a.batch, a.seed, rnd)
            raw_cache.clear()
            raw_cache[rnd] = (G.score(Hfix, Xr, Yr, enc)["first"][:, 0, :],
                              T.target_offsets(Yr))
        return raw_cache[rnd]

    curve, best_ho = [], []
    for rnd in range(a.rounds + 1):
        rw, tg = raw_for(rnd)
        fit = np.array([mse_of(p["aff_a"] * rw + p["aff_b"], tg) for p in pool])
        i = int(np.argmin(fit))
        ho = mse_of(pool[i]["aff_a"] * raw_v + pool[i]["aff_b"], tgt_v)
        curve.append(dict(rnd=rnd, train_mse=float(fit.min()), heldout_mse=ho))
        best_ho.append(ho)
        order = np.argsort(fit)
        keep = [pool[j] for j in order[:a.pool // 4]]
        pool = list(keep)
        while len(pool) < a.pool:
            p = keep[int(rng.integers(0, len(keep)))]
            pool.append({**p,
                         "aff_a": np.clip(p["aff_a"] + rng.normal(0, G.SIGMA_A, G.N_OUT),
                                          -G.AFF_A_LIM, G.AFF_A_LIM),
                         "aff_b": np.clip(p["aff_b"] + rng.normal(0, G.SIGMA_B, G.N_OUT),
                                          -G.AFF_B_LIM, G.AFF_B_LIM)})

    within = next((c["rnd"] for c in curve if c["heldout_mse"] <= target * 1.01), None)
    R["adjusts_fast"] = dict(
        frozen_net="grow-norm leader", target_analytic_train_fit=target,
        rounds_to_within_1pct=within,
        milestones={str(k): curve[k]["heldout_mse"] for k in (0, 25, 50, 100, 200)
                    if k < len(curve)},
        final_heldout=curve[-1]["heldout_mse"], curve=curve,
        ok=bool(within is not None))
    print("c ADJUSTS FAST", json.dumps(T.jsonable(
        {k: v for k, v in R["adjusts_fast"].items() if k != "curve"}), indent=1), flush=True)

    R["all_ok"] = bool(R["identity"]["ok"] and R["invariants"]["ok"]
                       and R["adjusts_fast"]["ok"])
    print(f"\nALL_OK={R['all_ok']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
