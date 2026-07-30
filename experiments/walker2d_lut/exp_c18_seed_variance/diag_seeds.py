"""exp_c18 — WHY do the seeds differ? Three diagnostics, numpy only (#75). MJX venv.

Reporting a seed sd and stopping there would leave the useful part unsaid. Each block below
answers one question that could plausibly explain a large spread, and each is falsifiable:

  (a) HAS THE ADDRESSING CONVERGED BY 10k? Measured from the --snap-every 500 snapshots as
      per-500-iteration movement, early vs late. Two units, because they disagree in an
      informative way: per-row ROTATION (geometry of w) and BIT-FLIP RATE on real
      observations (what addressing actually is). A hyperplane can drift several degrees and
      address identically; it can also barely move and flip many bits if it sits near the
      data. Late movement that is still comparable to early movement means 10k iterations is
      simply not enough training, and the "seed variance" is partly unconverged runs.

  (b) DEAD / COLLAPSED ROWS. Two different coverage notions, deliberately both:
        - TRAINING coverage: fraction of the 2,048 rows that ever received a gradient
          (the row_updates histogram carried in the snapshots).
        - DEPLOYED coverage: how many distinct rows the FINAL policy actually addresses on
          real observations. A table that addresses 2 of its 64 rows has thrown away its
          capacity regardless of what training touched.
      A row that trained but is never addressed at deployment is wasted capacity; a row
      addressed but never trained is emitting its init value, which is worse.

  (c) DOES THE SCORE TRACK ANY INIT PROPERTY? The init is a deterministic function of the
      seed (canonical_full_coverage anchor draw), so if seed 4 is bad, its init can be
      interrogated. The candidate properties are all about whether the initial bits are
      usable: a bit whose sign test is ~always true on real data carries no information, and
      a table full of such bits starts with a collapsed address space.
      n = 6, so correlations here are suggestive at best and are reported with that stated.

Read-only: loads checkpoints and snapshots, trains nothing, writes one JSON.
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
C03 = os.path.join(D, "exp_c03_distillation")
SEEDS = (0, 1, 2, 3, 4, 5)
N_OBS = 20000
OBS_SAMPLE_SEED = 0     # fixed, so every seed is judged on the identical observations
DEGENERATE = 0.02      # a bit true <2% or >98% of the time carries ~no address information
COLLAPSED_ROWS = 4     # a table addressing <=4 of its 64 rows has effectively collapsed


def load_obs():
    """A RANDOM sample of the 4.0M-row dataset, not the first N rows.

    This is not fussiness. obs.npy is stored in collection order, so obs.npy[:N] is a
    single narrow window of trajectory: standardised, its per-dim std comes out around
    0.001-0.03 instead of 1, and 97% of the sign tests are then constant across the
    sample -- which makes every bit look dead and every table look collapsed to ~1 of
    64 rows. Sampling at random over the whole file recovers std ~= 1 and P(bit) in
    [0.39, 0.79]. (exp_c14/diag_hyperplane_movement.py reads obs.npy[:2000] and so has
    the narrow-window flaw; its bit-flip percentages should be read as measured on an
    unrepresentative slice. This file supersedes them for this config.)
    """
    st = json.load(open(os.path.join(C03, "dataset_stats.json")))
    om = np.asarray(st["obs_mean"], np.float64)
    osd = np.asarray(st["obs_std"], np.float64)
    obs = np.load(os.path.join(C03, "obs.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(OBS_SAMPLE_SEED).choice(
        len(obs), min(N_OBS, len(obs)), replace=False))
    return (np.asarray(obs[idx], np.float64) - om) / (osd + 1e-6)


def bits_and_rows(x, w, b):
    """sign tests and the big-endian row index, matching _hard_index in jax_lut_grad."""
    a = np.einsum("bd,tnd->btn", x, w) + b[None]
    bit = a > 0
    nap = w.shape[1]
    powers = (2 ** np.arange(nap - 1, -1, -1)).astype(np.int64)
    return bit, (bit.astype(np.int64) * powers[None, None, :]).sum(-1)   # [B,T], [B,T]


def rotation_deg(w0, w1):
    """Per-ROW angle. Rows are the objects that define a bit, so a global tensor angle
    would average away exactly the structure being asked about."""
    a_ = w0.reshape(-1, w0.shape[-1])
    b_ = w1.reshape(-1, w1.shape[-1])
    cos = ((a_ * b_).sum(-1)
           / (np.linalg.norm(a_, axis=-1) * np.linalg.norm(b_, axis=-1) + 1e-12))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    x = load_obs()
    print(f"{x.shape[0]} standardised observations from the distillation dataset")

    res_path = os.path.join(HERE, "seed_variance_results.json")
    scores = {}
    if os.path.exists(res_path):
        scores = {int(s["seed"]): s["mean"]
                  for s in json.load(open(res_path))["seeds"]}

    per_seed = []
    for s in SEEDS:
        ck = os.path.join(C09, f"lut_sac_c18_seed{s}_actor.npz")
        sn = os.path.join(C09, f"lut_sac_c18_seed{s}_snaps.npz")
        if not (os.path.exists(ck) and os.path.exists(sn)):
            print(f"  (seed {s}: missing checkpoint or snapshots — skipped)")
            continue
        z, zs = np.load(ck), np.load(sn)
        wf, bf = np.asarray(z["w"], np.float64), np.asarray(z["b"], np.float64)
        its = np.asarray(zs["iters"])
        W, B = np.asarray(zs["w"], np.float64), np.asarray(zs["b"], np.float64)
        RU = np.asarray(zs["row_updates"])
        n_tables, nap = wf.shape[0], wf.shape[1]
        K = 2 ** nap

        # ---- (a) movement per 500 iters, snapshot to snapshot ------------------
        seg = []
        for i in range(1, len(its)):
            d_it = int(its[i] - its[i - 1])
            rot = rotation_deg(W[i - 1], W[i])
            f0, _ = bits_and_rows(x, W[i - 1], B[i - 1])
            f1, _ = bits_and_rows(x, W[i], B[i])
            seg.append(dict(a=int(its[i - 1]), b=int(its[i]),
                            rot_per500=float(rot.mean() * 500 / d_it),
                            flip_per500=float((f0 != f1).mean() * 500 / d_it)))
        # "early" = the first 2,000 iters of actual learning; "late" = the last 2,000
        early = [g for g in seg if g["b"] <= 2500 and g["a"] >= 500]
        late = [g for g in seg if g["a"] >= its[-1] - 2000]
        e_rot = float(np.mean([g["rot_per500"] for g in early])) if early else float("nan")
        l_rot = float(np.mean([g["rot_per500"] for g in late])) if late else float("nan")
        e_flip = float(np.mean([g["flip_per500"] for g in early])) if early else float("nan")
        l_flip = float(np.mean([g["flip_per500"] for g in late])) if late else float("nan")

        # ---- (b) coverage: trained vs actually addressed -----------------------
        train_cov = float((RU[-1] > 0).mean())
        bit_f, rows_f = bits_and_rows(x, wf, bf)
        used = np.zeros((n_tables, K), bool)
        for t in range(n_tables):
            used[t, np.unique(rows_f[:, t])] = True
        rows_used = used.sum(1)                       # distinct rows per table, of K
        deployed_cov = float(used.mean())
        collapsed = int((rows_used <= COLLAPSED_ROWS).sum())
        # rows that got gradient but are never addressed = wasted capacity;
        # rows addressed but never trained = still emitting their init value.
        trained = RU[-1] > 0
        wasted = float((trained & ~used).mean())
        untrained_used = float((~trained & used).mean())
        # address entropy, in bits out of nap: how much of the address space is live
        ent = []
        for t in range(n_tables):
            c = np.bincount(rows_f[:, t], minlength=K).astype(np.float64)
            p = c[c > 0] / c.sum()
            ent.append(-(p * np.log2(p)).sum())
        ent = float(np.mean(ent))

        # ---- (c) init properties (deterministic in the seed) ------------------
        bit_i, rows_i = bits_and_rows(x, W[0], B[0])
        p_true = bit_i.mean(0)                        # [T, nap] fraction of positives
        init_degen = int(((p_true < DEGENERATE) | (p_true > 1 - DEGENERATE)).sum())
        init_balance = float(np.abs(0.5 - p_true).mean())
        used_i = np.zeros((n_tables, K), bool)
        for t in range(n_tables):
            used_i[t, np.unique(rows_i[:, t])] = True
        init_cov = float(used_i.mean())

        per_seed.append(dict(
            seed=s, score=scores.get(s), n_tables=n_tables, nap=nap, K=K,
            early_rot_per500=e_rot, late_rot_per500=l_rot,
            early_flip_per500=e_flip, late_flip_per500=l_flip,
            movement_ratio=(l_rot / e_rot if e_rot else float("nan")),
            flip_ratio=(l_flip / e_flip if e_flip else float("nan")),
            train_coverage=train_cov, deployed_coverage=deployed_cov,
            mean_rows_used=float(rows_used.mean()), min_rows_used=int(rows_used.min()),
            collapsed_tables=collapsed, wasted_frac=wasted,
            untrained_but_used_frac=untrained_used, address_entropy_bits=ent,
            init_degenerate_bits=init_degen, init_bit_imbalance=init_balance,
            init_coverage=init_cov, segments=seg))
        print(f"  seed {s}: parsed {len(its)} snapshots ({its[0]}..{its[-1]})")

    if not per_seed:
        print("no seeds ready — nothing to diagnose")
        return

    # ---- report --------------------------------------------------------------
    print("\n=== (a) IS THE ADDRESSING STILL MOVING AT 10k? (per 500 iters) ===")
    print(f"{'seed':>5}{'rot early':>11}{'rot late':>10}{'late/early':>12}"
          f"{'flip early':>12}{'flip late':>11}{'late/early':>12}")
    for r in per_seed:
        print(f"{r['seed']:>5}{r['early_rot_per500']:>10.2f}°{r['late_rot_per500']:>9.2f}°"
              f"{r['movement_ratio']:>12.2f}{100*r['early_flip_per500']:>11.2f}%"
              f"{100*r['late_flip_per500']:>10.2f}%{r['flip_ratio']:>12.2f}")
    # nan-safe: a partially-trained seed has no late window yet and must not poison the
    # aggregate into a silent nan (which would fall through to a "converged" verdict).
    with np.errstate(invalid="ignore"):
        mr = float(np.nanmean([r["movement_ratio"] for r in per_seed]))
        fr = float(np.nanmean([r["flip_ratio"] for r in per_seed]))
    if not np.isfinite(fr):
        conv = ("INCONCLUSIVE: not enough snapshots to compare early against late "
                "movement. Needs a run with --snap-every covering both windows.")
    elif fr > 0.5:
        conv = (f"NOT CONVERGED. Late bit-flip rate is {fr:.2f}x the early rate, so the "
                f"addressing is still being rewritten in the final 2,000 iterations. "
                f"10k iters is a cut-off, not a resting point -- part of the seed spread "
                f"is runs stopped at different points of an ongoing search.")
    elif fr > 0.15:
        conv = (f"PARTLY CONVERGED. Late movement is {fr:.2f}x early: the addressing has "
                f"slowed markedly but has not stopped. Longer training would likely still "
                f"move the numbers.")
    else:
        conv = (f"CONVERGED. Late movement is only {fr:.2f}x early -- the addressing has "
                f"settled well before 10k, so the seed spread is not an artefact of "
                f"stopping early. It reflects genuinely different solutions.")
    print(f"\n  {conv}")

    print("\n=== (b) COVERAGE AND DEAD ROWS ===")
    print(f"{'seed':>5}{'score':>9}{'trained':>9}{'deployed':>10}{'rows/table':>12}"
          f"{'min':>5}{'collapsed':>11}{'wasted':>9}{'entropy':>9}")
    for r in per_seed:
        sc = f"{r['score']:.0f}" if r["score"] is not None else "n/a"
        print(f"{r['seed']:>5}{sc:>9}{100*r['train_coverage']:>8.1f}%"
              f"{100*r['deployed_coverage']:>9.1f}%{r['mean_rows_used']:>9.1f}/{r['K']}"
              f"{r['min_rows_used']:>5}{r['collapsed_tables']:>8}/{r['n_tables']}"
              f"{100*r['wasted_frac']:>8.1f}%{r['address_entropy_bits']:>8.2f}b")
    print(f"  (entropy is out of {per_seed[0]['nap']} bits; 'wasted' = rows that received "
          f"gradient but are never addressed at deployment)")
    ub = max(r["untrained_but_used_frac"] for r in per_seed)
    if ub > 0.001:
        print(f"  WARNING: up to {100*ub:.1f}% of rows are addressed at deployment but "
              f"never received a gradient -- those emit their init value.")

    print("\n=== (c) DOES THE SCORE TRACK ANY INIT PROPERTY? ===")
    have = [r for r in per_seed if r["score"] is not None]
    corrs = {}
    if len(have) >= 4:
        y = np.array([r["score"] for r in have], np.float64)
        cand = dict(init_degenerate_bits="degenerate bits at init",
                    init_bit_imbalance="mean |0.5 - P(bit)| at init",
                    init_coverage="rows addressed at init",
                    deployed_coverage="rows addressed at 10k",
                    late_flip_per500="late bit-flip rate",
                    train_coverage="training row coverage")
        print(f"{'property':<34}{'pearson r':>11}{'spearman':>10}    values")
        for k, lab in cand.items():
            v = np.array([r[k] for r in have], np.float64)
            if v.std() < 1e-12:
                print(f"{lab:<34}{'constant':>11}{'':>10}    {v[0]:.4g} for every seed")
                corrs[k] = None
                continue
            pr = float(np.corrcoef(v, y)[0, 1])
            rk = lambda a: np.argsort(np.argsort(a)).astype(np.float64)
            sr = float(np.corrcoef(rk(v), rk(y))[0, 1])
            corrs[k] = dict(pearson=pr, spearman=sr,
                            values=[float(t) for t in v])
            print(f"{lab:<34}{pr:>11.3f}{sr:>10.3f}    "
                  + " ".join(f"{t:.3g}" for t in v))
        print(f"\n  n = {len(have)}. At n=6 the 5% significance threshold for |r| is about "
              f"0.81, so treat anything below that as a hint to test at more seeds, not "
              f"a finding. Sign and magnitude are still worth recording.")
        best = max((k for k in corrs if corrs[k]), key=lambda k: abs(corrs[k]["pearson"]),
                   default=None)
        if best:
            print(f"  strongest: {cand[best]} (r = {corrs[best]['pearson']:+.3f})")

    json.dump(dict(seeds=per_seed, convergence_verdict=conv,
                   mean_rot_ratio=mr, mean_flip_ratio=fr, correlations=corrs,
                   n_obs=int(x.shape[0]), degenerate_threshold=DEGENERATE,
                   collapsed_row_threshold=COLLAPSED_ROWS),
              open(os.path.join(HERE, "seed_diagnostics.json"), "w"), indent=1)
    print("\nwrote seed_diagnostics.json")


if __name__ == "__main__":
    main()
