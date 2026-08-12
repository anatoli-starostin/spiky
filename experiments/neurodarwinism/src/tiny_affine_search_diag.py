"""exp012: why does evolution leave 13.5 MSE on the table in its OWN scale/shift genes?

The champion's evolved diagonal scores 48.2 where the best diagonal for that same network is
34.7. The operator is not broken -- on a frozen net it reaches the optimum in ~100 rounds. So
the question is what makes it fail in a real run.

The hypothesis this measures: a single affine mutation changes the MSE by far LESS than the
minibatch changes it, so selection literally cannot see the step. Selection compares genomes
scored on the SAME batch, so what matters is the signal a step produces against the spread of
the population -- but a parent and its mutated child are also separated by every other
mutation they carry, and by the batch resample between the rounds in which they are judged.

  1 SIGNAL   distribution of |dMSE| from ONE affine step, at the current sigmas
  2 NOISE    the same genome scored on different minibatches -- the floor a step must clear
  3 SWEEP    what sigma actually makes the step visible, and does a bigger sigma still
             converge on a frozen net (or does it just thrash)?
"""
import argparse
import json

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    G.QUANTIZED = a.quantized
    from tiny_grow_evolve import load_ckpt
    pool, ewma, *_ = load_ckpt(a.ckpt)
    fin = np.where(np.isfinite(ewma))[0]
    g = pool[int(fin[np.argmin(ewma[fin])])]

    _, _, Xp, Yp, Xv, Yv = load(256, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    rng = np.random.default_rng(a.seed)
    H = G.build([g], device=a.device)

    # raw first-spike offsets on a few different minibatches -- the forward pass is what costs,
    # so cache it and do all the affine arithmetic in numpy
    raws, tgts = [], []
    for r in range(12):
        Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, 100 + r)
        raws.append(G.score(H, Xb, Yb, enc)["first"][:, 0, :])
        tgts.append(T.target_offsets(Yb))
    rv = G.score(H, Xv, Yv, enc)["first"][:, 0, :]
    tv = T.target_offsets(Yv)
    chance = T.constant_baseline(Yv)

    def mse(aa, bb, raw, tgt):
        return float((((aa * raw + bb) - tgt) ** 2).mean())

    a0, b0 = G.affine_of(g)
    at, bt = G.analytic_affine(raws[0], tgts[0])

    R = dict(chance=chance, evolved=mse(a0, b0, rv, tv), best_diag=mse(at, bt, rv, tv),
             evolved_a=a0.tolist(), evolved_b=b0.tolist(),
             best_a=at.tolist(), best_b=bt.tolist(),
             sigma_a=G.SIGMA_A, sigma_b=G.SIGMA_B, p_affine=0.25)

    # ---- 1 SIGNAL: one affine step, at the current sigmas, on a fixed batch
    base = mse(a0, b0, raws[0], tgts[0])
    d = []
    for _ in range(a.trials):
        ma = rng.random(T.N_OUT) < 0.25
        mb = rng.random(T.N_OUT) < 0.25
        aa = np.clip(a0 + ma * rng.normal(0, G.SIGMA_A, T.N_OUT), -G.AFF_A_LIM, G.AFF_A_LIM)
        bb = np.clip(b0 + mb * rng.normal(0, G.SIGMA_B, T.N_OUT), -G.AFF_B_LIM, G.AFF_B_LIM)
        d.append(mse(aa, bb, raws[0], tgts[0]) - base)
    d = np.array(d)
    R["signal_one_step"] = dict(mean_abs=float(np.abs(d).mean()), sd=float(d.std()),
                                p90_abs=float(np.percentile(np.abs(d), 90)))

    # ---- 2 NOISE: the SAME genome, different minibatches
    per_batch = np.array([mse(a0, b0, raws[i], tgts[i]) for i in range(len(raws))])
    R["noise_minibatch"] = dict(sd=float(per_batch.std()),
                                range=[float(per_batch.min()), float(per_batch.max())],
                                mean=float(per_batch.mean()))
    R["signal_to_noise"] = R["signal_one_step"]["mean_abs"] / R["noise_minibatch"]["sd"]

    # ---- 3 SWEEP sigma, and check convergence on a frozen net with REAL batch resampling
    def evolve_affine(sa, sb, rounds=400, pool_n=64, p=0.25):
        P = [(a0.copy(), b0.copy())]
        for _ in range(pool_n - 1):
            P.append((np.clip(a0 + rng.normal(0, sa, T.N_OUT), -4, 4),
                      np.clip(b0 + rng.normal(0, sb, T.N_OUT), -64, 64)))
        best_ho = []
        for r in range(rounds):
            raw, tgt = raws[r % len(raws)], tgts[r % len(tgts)]      # resampled each round
            f = np.array([mse(x[0], x[1], raw, tgt) for x in P])
            keep = [P[i] for i in np.argsort(f)[:pool_n // 4]]
            best_ho.append(mse(keep[0][0], keep[0][1], rv, tv))
            P = list(keep)
            while len(P) < pool_n:
                q = keep[int(rng.integers(0, len(keep)))]
                ma, mb = rng.random(T.N_OUT) < p, rng.random(T.N_OUT) < p
                P.append((np.clip(q[0] + ma * rng.normal(0, sa, T.N_OUT), -4, 4),
                          np.clip(q[1] + mb * rng.normal(0, sb, T.N_OUT), -64, 64)))
        return best_ho

    R["sigma_sweep"] = {}
    for sa, sb in ((0.05, 0.308), (0.10, 1.0), (0.15, 1.5), (0.25, 3.0), (0.4, 6.0)):
        h = evolve_affine(sa, sb)
        R["sigma_sweep"][f"a{sa}_b{sb}"] = dict(
            sigma_a=sa, sigma_b=sb, at_50=h[50], at_100=h[100], at_200=h[200],
            final=h[-1], best=float(np.min(h)))
        print(f"  sigma_a {sa:4.2f} sigma_b {sb:5.2f}   r50 {h[50]:7.2f}  r100 {h[100]:7.2f}"
              f"  r200 {h[200]:7.2f}  final {h[-1]:7.2f}  best {np.min(h):7.2f}", flush=True)

    print(f"\nevolved {R['evolved']:.2f} | best diagonal {R['best_diag']:.2f} | "
          f"chance {chance:.2f}")
    print(f"one affine step moves MSE by {R['signal_one_step']['mean_abs']:.4f} on average; "
          f"minibatch noise sd is {R['noise_minibatch']['sd']:.4f}")
    print(f"SIGNAL-TO-NOISE = {R['signal_to_noise']:.3f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(R), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
