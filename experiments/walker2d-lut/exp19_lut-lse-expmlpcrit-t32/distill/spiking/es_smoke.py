"""Phase-1 gates: continuous-weight round trip, encoder, null baseline, isolation, ES, arm B."""
import argparse

import numpy as np
import torch

import es_harness as H
from es_harness import (D_MAX, D_MIN, N_EXC, N_INH, N_IN, N_OUT, N_TICKS, T_IN,
                        LatencyEncoder, build, fitness, kendall_tau_b, mutate,
                        random_genome, reservoir_wiring, run_episode, verify_round_trip)


def load(n, seed=0, n_val=2000):
    """-> fixed reference batch (X,Y), the whole training pool, and a FIXED val set.

    SEEDING SCHEME, so runs are reproducible:
      * the fixed reference batch (used for the null baseline and the smoke gates) is
        drawn with numpy default_rng(seed).
      * the FIXED val set is drawn from the last 4000 samples with default_rng(seed + 1).
      * each generation g draws its own training batch with default_rng(seed*1000 + g)
        — see sample_batch(). One draw per generation, SHARED across the population, so
        candidates stay perfectly paired within a generation.
    """
    Z = np.load(H.NPZ)
    N = Z["x_norm"].shape[0]
    n_tr = N - 4000
    Xp = Z["x_norm"][:n_tr].astype(np.float64)
    Yp = Z["y_action_mean"][:n_tr].astype(np.float64)
    ref = np.random.default_rng(seed).choice(n_tr, n, replace=False)
    vi = np.random.default_rng(seed + 1).choice(np.arange(n_tr, N), n_val, replace=False)
    return (Xp[ref], Yp[ref], Xp, Yp,
            Z["x_norm"][vi].astype(np.float64), Z["y_action_mean"][vi].astype(np.float64))


def sample_batch(Xp, Yp, n, seed, gen):
    """Fresh training batch for generation `gen`, deterministic in (seed, gen)."""
    idx = np.random.default_rng(seed * 1000 + gen).choice(Xp.shape[0], n, replace=False)
    return Xp[idx], Yp[idx], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-max", type=float, default=10.0)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--res-scale", type=float, default=1.0)
    ap.add_argument("--skip-armb", action="store_true")
    ap.add_argument("--stdp-seconds", type=int, default=250)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    res = reservoir_wiring(np.random.default_rng(1234))
    X, Y, Xpool, Ypool, Xval, Yval = load(a.batch, a.seed, a.n_val)
    enc = LatencyEncoder(Xpool)
    print(f"  train pool {Xpool.shape[0]:,}  fixed reference batch {X.shape[0]}  "
          f"fixed val {Xval.shape[0]}")
    print(f"device {dev}  batch {a.batch}  w_max {a.w_max}  res_scale {a.res_scale}")

    # ---- GATE 0: Dale's law on the evolved genome ---------------------------------
    print("\n=== GATE 0: Dale's law (evolved weights must be >= 0) ===")
    gd = random_genome(np.random.default_rng(0), a.w_max)
    for k in ("in_w", "out_w"):
        v = gd[k]
        print(f"  init {k:6s}: range [{v.min():.3f}, {v.max():.3f}]  "
              f"negative fraction {float((v < 0).mean()):.4f}")
    md = gd
    for i in range(20):
        md = mutate(md, np.random.default_rng(i), a.w_max)
    dale_ok = True
    for k in ("in_w", "out_w"):
        v = md[k]
        neg = float((v < 0).mean())
        dale_ok &= neg == 0.0
        print(f"  after 20 mutations {k:6s}: range [{v.min():.3f}, {v.max():.3f}]  "
              f"negative fraction {neg:.4f}  at-zero fraction {float((v == 0).mean()):.4f}")
    print(f"  DALE GATE: {'PASS' if dale_ok else 'FAIL'}")

    # ---- GATE 1: continuous-weight round trip, CUDA and CPU -----------------------
    print("\n=== GATE 1: continuous explicit weights round-trip ===")
    g0 = random_genome(rng, a.w_max)
    for d in (["cuda", "cpu"] if dev == "cuda" else ["cpu"]):
        h = build([g0], res, d, res_scale=a.res_scale)
        r = verify_round_trip(h)
        E = r["n_requested"]
        print(f"  {d:4s}: exported {r['n_exported']:,} for {E:,} edges | "
              f"weights {r['weights_ok']}/{E}  delays {r['delays_ok']}/{E}  "
              f"missing {r['missing']}  -> "
              f"{'EXACT' if r['weights_ok'] == E and r['delays_ok'] == E and not r['missing'] else 'FAILED'}")
        if d == dev:
            h1 = h
        else:
            del h

    # ---- encoder ------------------------------------------------------------------
    tk = enc(X)
    print(f"\nencoder: input ticks min {tk.min()} max {tk.max()} "
          f"distinct {len(np.unique(tk))} of {T_IN} "
          f"(was 11 of 32 with the inherited LUT fit)")

    # ---- GATE 2: single candidate --------------------------------------------------
    print("\n=== GATE 2: single candidate ===")
    from spiky.spnet.spnet import NeuronDataType
    sp, ids = h1["spnet"], h1["ids"]
    first, R = run_episode(h1, X, enc, a.current)
    Rin = sp.export_neuron_data(torch.tensor(ids[2], dtype=torch.int32, device=dev),
                                X.shape[0], NeuronDataType.Spike, 0, N_TICKS - 1).cpu().numpy()
    fired_in = Rin.any(-1)
    ft = np.where(fired_in, Rin.argmax(-1), -1)
    Rexc = sp.export_neuron_data(torch.tensor(ids[0], dtype=torch.int32, device=dev),
                                 X.shape[0], NeuronDataType.Spike, 0, N_TICKS - 1).cpu().numpy()
    print(f"  input neurons fired {fired_in.mean()*100:.1f}%, median lag "
          f"{np.median((ft - tk)[fired_in]) if fired_in.any() else float('nan'):.1f} ticks")
    print(f"  reservoir exc: {Rexc.sum()/X.shape[0]:.1f} spikes/sample, "
          f"{Rexc.any(-1).sum()/X.shape[0]/N_EXC*100:.1f}% of exc neurons recruited")
    of = first < N_TICKS
    print(f"  output neurons fired {of.mean()*100:.1f}%, first-spike ticks "
          f"{first[of].min() if of.any() else -1:.0f}-{first[of].max() if of.any() else -1:.0f}")

    # ---- GATE 3: null baseline ------------------------------------------------------
    print("\n=== GATE 3: tau-b NULL baseline ===")
    tau_real = kendall_tau_b(-first[:, 0, :], Y)
    nulls = []
    for k in range(200):
        perm = np.random.default_rng(k).permutation(Y.shape[0])
        nulls.append(kendall_tau_b(-first[:, 0, :], Y[perm]).mean())
    nulls = np.array(nulls)
    within = [kendall_tau_b(-first[:, 0, :],
                            Y[:, np.random.default_rng(k).permutation(N_OUT)]).mean()
              for k in range(200)]
    within = np.array(within)
    se = tau_real.std() / np.sqrt(len(tau_real))
    print(f"  random-wiring tau-b      {tau_real.mean():+.4f}  (SE {se:.4f})")
    print(f"  label-shuffle null       {nulls.mean():+.4f}  sd {nulls.std():.4f}  "
          f"[{np.percentile(nulls,2.5):+.4f}, {np.percentile(nulls,97.5):+.4f}]")
    print(f"  within-sample dim-shuffle{within.mean():+.4f}  sd {within.std():.4f}")
    z = (tau_real.mean() - nulls.mean()) / max(nulls.std(), 1e-9)
    print(f"  => random wiring sits {z:+.2f} sd from the label-shuffle null")
    null_mu = float(nulls.mean())

    # ---- GATE 4: isolation ----------------------------------------------------------
    print(f"\n=== GATE 4: isolation, {a.pop} packed vs solo ===")
    gs = [g0] + [random_genome(rng, a.w_max) for _ in range(a.pop - 1)]
    hp = build(gs, res, dev, res_scale=a.res_scale)
    fp, Rp = run_episode(hp, X, enc, a.current)
    allok = True
    for c in range(a.pop):
        hs = build([gs[c]], res, dev, res_scale=a.res_scale)
        fs, Rs = run_episode(hs, X, enc, a.current)
        ok = np.array_equal(Rp[:, c], Rs[:, 0])
        print(f"  candidate {c}: raster identical {ok} "
              f"(differing {int((Rp[:, c] != Rs[:, 0]).sum())})")
        allok &= ok
        del hs
    print(f"  ISOLATION GATE: {'PASS' if allok else 'FAIL'}")

    # ---- GATE 5: tiny ES ------------------------------------------------------------
    if a.gens and allok:
        print(f"\n=== GATE 5: {a.gens} generations, pop {a.pop}, batch {a.batch} "
              f"RESAMPLED each gen (null {null_mu:+.4f}, on the fixed reference batch) ===")
        print(f"  {'gen':>3} {'idx[:4]':>22} {'train best':>11} {'train mean':>11} "
              f"{'elite val':>10} {'train-null':>11} {'val-null':>9}")
        elite, cur_val = gs[0], None
        pop = gs
        for gen in range(0, a.gens + 1):
            Xb, Yb, idx = sample_batch(Xpool, Ypool, a.batch, a.seed, gen)
            if gen > 0:
                pop = [elite] + [mutate(elite, rng, a.w_max) for _ in range(a.pop - 1)]
            hk = build(pop, res, dev, res_scale=a.res_scale)
            fk, _, _ = fitness(hk, Xb, Yb, enc, a.current)      # fresh batch, shared by pop
            elite = pop[int(fk.argmax())]
            # score THAT elite on the fixed held-out set
            hv = build([elite], res, dev, res_scale=a.res_scale)
            fv, _, _ = fitness(hv, Xval, Yval, enc, a.current)
            cur_val = float(fv[0])
            print(f"  {gen:>3} {str(idx[:4].tolist()):>22} {fk.max():>+11.4f} "
                  f"{fk.mean():>+11.4f} {cur_val:>+10.4f} {fk.max()-null_mu:>+11.4f} "
                  f"{cur_val-null_mu:>+9.4f}")
            del hk, hv

    # ---- ARM B: edge-exact import of the STDP reservoir -----------------------------
    if not a.skip_armb:
        print("\n=== ARM B: edge-exact import of an STDP-pretrained reservoir ===")
        import arm_b
        arm_b.run(dev, res, g0, enc, X, Y, a)


if __name__ == "__main__":
    main()
