"""exp_c18 — WHY does seed 4 jackpot? (#75). Runs in the SPIKY venv (numpy + matplotlib).

Five seeds land at 4112 +/- 160; seed 4 lands at 5287. Determinism (exp_c17) means this is
a real, repeatable property of that seed and not a float-reassociation accident, so it has
a mechanism worth finding.

Four questions, each with a concrete measurement, and each capable of coming back "no":

  1. ADDRESSING DYNAMICS per seed -- did seed 4's addressing converge earlier, later, or
     differently? Segment-wise rotation and bit-flip rate across the whole run, plus the
     cumulative distance travelled from init.
  2. FINAL ROW USAGE per seed -- distribution over the 32 tables, not just the mean, since
     a mean hides "one table collapsed to 14 rows".
  3. INIT DIFFERENCES. Under anchor_pairs init every seed starts from the SAME structure
     (w = e_a - e_b, b = 0) and differs only in WHICH pairs were drawn, so the starting
     points can be compared exactly: pair-set overlap between seeds, how redundant a
     table's 6 bits are on real data, and how much of the address space each init reaches.
     The init table weights are compared too (from dump_table_init.py).
  4. CORRELATION of score with init and dynamics properties.

Honest limit stated up front: CRITIC LOSS IS NOT AVAILABLE. lut_sac.py computes q_loss but
persists only iter / env_steps / mjx_return / row_coverage / alpha / elapsed. Adding it is
a one-line change for future runs; it cannot be recovered for these six without retraining.
alpha is the dynamics signal that IS logged, and it is used instead.
"""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
C03 = os.path.join(D, "exp_c03_distillation")
SEEDS = (0, 1, 2, 3, 4, 5)
STAR = 4                      # the outlier under investigation
N_OBS = 20000
OBS_SAMPLE_SEED = 0
TABLE_INIT = "/tmp/c18_table_init.npz"
FIG = os.path.join(HERE, "seed4_diagnostics.png")


def load_obs():
    st = json.load(open(os.path.join(C03, "dataset_stats.json")))
    om = np.asarray(st["obs_mean"], np.float64)
    osd = np.asarray(st["obs_std"], np.float64)
    obs = np.load(os.path.join(C03, "obs.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(OBS_SAMPLE_SEED).choice(
        len(obs), min(N_OBS, len(obs)), replace=False))
    return (np.asarray(obs[idx], np.float64) - om) / (osd + 1e-6)


def bits_rows(x, w, b):
    a = np.einsum("bd,tnd->btn", x, w) + b[None]
    bit = a > 0
    powers = (2 ** np.arange(w.shape[1] - 1, -1, -1)).astype(np.int64)
    return bit, (bit.astype(np.int64) * powers[None, None, :]).sum(-1)


def per_row_angle(w0, w1):
    a_ = w0.reshape(-1, w0.shape[-1]); b_ = w1.reshape(-1, w1.shape[-1])
    cos = ((a_ * b_).sum(-1)
           / (np.linalg.norm(a_, axis=-1) * np.linalg.norm(b_, axis=-1) + 1e-12))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def pair_set(w0):
    """The anchor pairs the init drew, as an unordered {a,b} per (table, bit)."""
    pairs = []
    for t in range(w0.shape[0]):
        for n in range(w0.shape[1]):
            ia = np.where(w0[t, n] > 0.5)[0]
            ib = np.where(w0[t, n] < -0.5)[0]
            pairs.append((int(ia[0]), int(ib[0])) if len(ia) and len(ib) else (-1, -1))
    return pairs


def corr_within_table(bit):
    """Mean |pairwise correlation| between the nap bits of a table, on real data.
    High redundancy means several bits say the same thing and the 2^nap address space
    is not actually reachable -- the cleanest single number for 'how much addressing
    does this init really have'."""
    T, nap = bit.shape[1], bit.shape[2]
    out = []
    for t in range(T):
        v = bit[:, t, :].astype(np.float64)
        v = v - v.mean(0)
        sd = v.std(0) + 1e-12
        c = (v.T @ v) / (len(v) * sd[:, None] * sd[None, :])
        iu = np.triu_indices(nap, 1)
        out.append(np.abs(c[iu]).mean())
    return float(np.mean(out))


def main():
    x = load_obs()
    scores = {int(s["seed"]): s["mean"]
              for s in json.load(open(os.path.join(
                  HERE, "seed_variance_results.json")))["seeds"]}
    tinit = np.load(TABLE_INIT) if os.path.exists(TABLE_INIT) else None

    R = {}
    for s in SEEDS:
        z = np.load(os.path.join(C09, f"lut_sac_c18_seed{s}_actor.npz"))
        zs = np.load(os.path.join(C09, f"lut_sac_c18_seed{s}_snaps.npz"))
        hist = json.load(open(os.path.join(C09, f"lut_sac_c18_seed{s}.json")))["history"]
        its = np.asarray(zs["iters"])
        W = np.asarray(zs["w"], np.float64); B = np.asarray(zs["b"], np.float64)
        wf = np.asarray(z["w"], np.float64); bf = np.asarray(z["b"], np.float64)
        K = 2 ** wf.shape[1]

        # --- 1. dynamics: per-segment movement, normalised per 500 iters ----
        seg_it, seg_rot, seg_flip = [], [], []
        prev_bit, _ = bits_rows(x, W[0], B[0])
        for i in range(1, len(its)):
            d = int(its[i] - its[i - 1])
            bit_i, _ = bits_rows(x, W[i], B[i])
            seg_it.append(int(its[i]))
            seg_rot.append(float(per_row_angle(W[i - 1], W[i]).mean() * 500 / d))
            seg_flip.append(float((prev_bit != bit_i).mean() * 500 / d))
            prev_bit = bit_i
        seg_rot = np.array(seg_rot); seg_flip = np.array(seg_flip)
        cum_ang = float(per_row_angle(W[0], wf).mean())
        cum_dw = float(np.linalg.norm(wf - W[0]) / np.linalg.norm(W[0]))

        # --- 2. final row usage per table ----------------------------------
        bit_f, rows_f = bits_rows(x, wf, bf)
        used = np.zeros((wf.shape[0], K), bool)
        for t in range(wf.shape[0]):
            used[t, np.unique(rows_f[:, t])] = True
        ru = used.sum(1)

        # --- 3. init properties --------------------------------------------
        bit_i0, rows_i0 = bits_rows(x, W[0], B[0])
        used_i = np.zeros((wf.shape[0], K), bool)
        for t in range(wf.shape[0]):
            used_i[t, np.unique(rows_i0[:, t])] = True
        ru_i = used_i.sum(1)
        ps = pair_set(W[0])
        dims = np.zeros(wf.shape[-1], int)
        for a_, b_ in ps:
            dims[a_] += 1; dims[b_] += 1
        # duplicate pairs WITHIN a table: a repeated comparator is a wasted bit
        dup = 0
        nap = wf.shape[1]
        reuse = []
        for t in range(wf.shape[0]):
            tp = [tuple(sorted(p)) for p in ps[t * nap:(t + 1) * nap]]
            dup += nap - len(set(tp))
            # how often the 6 comparators of one table reuse the same obs dim. A table
            # built from 12 distinct dims tests 12 things; one built from 7 tests less.
            reuse.append(2 * nap - len({d for p in tp for d in p}))

        # --- 4. dynamics from the training history -------------------------
        ret = np.array([h["mjx_return"] for h in hist], np.float64)
        h_it = np.array([h["iter"] for h in hist], np.float64)
        alpha = np.array([(h["alpha"] if h["alpha"] is not None else np.nan)
                          for h in hist], np.float64)
        def first_at(th):
            w_ = np.where(ret >= th)[0]
            return int(h_it[w_[0]]) if len(w_) else -1

        R[s] = dict(
            seed=s, score=scores[s], seg_it=seg_it,
            seg_rot=seg_rot.tolist(), seg_flip=seg_flip.tolist(),
            cum_angle=cum_ang, cum_rel_dw=cum_dw,
            last_flip=float(seg_flip[-1]), first_flip=float(seg_flip[1]),
            flip_ratio=float(seg_flip[-1] / seg_flip[1]),
            total_flip=float(seg_flip.sum()),
            rows_used_mean=float(ru.mean()), rows_used_min=int(ru.min()),
            rows_used_max=int(ru.max()), rows_used_p25=float(np.percentile(ru, 25)),
            tables_under_32=int((ru < 32).sum()), tables_under_20=int((ru < 20).sum()),
            init_rows_mean=float(ru_i.mean()), init_rows_min=int(ru_i.min()),
            init_redundancy=corr_within_table(bit_i0),
            final_redundancy=corr_within_table(bit_f),
            init_dup_pairs=int(dup), init_dim_reuse=float(np.mean(reuse)),
            init_dim_gini=float(dims.std() / dims.mean()),
            init_dims_unused=int((dims == 0).sum()),
            pairs=ps,
            table_init_std=(float(tinit[f"weights_{s}"].std()) if tinit else None),
            table_init_norm=(float(np.linalg.norm(tinit[f"weights_{s}"]))
                             if tinit else None),
            ret=ret.tolist(), h_it=h_it.tolist(), alpha=alpha.tolist(),
            alpha_final=float(alpha[-1]), alpha_min=float(np.nanmin(alpha)),
            iters_to_2000=first_at(2000), iters_to_4000=first_at(4000),
            final_ret=float(ret[-1]))
        print(f"  seed {s}: parsed")

    # ---------------- report -------------------------------------------------
    def col(k, fmt="{:.3f}"):
        return "  ".join(fmt.format(R[s][k]) if R[s][k] is not None else "n/a"
                         for s in SEEDS)

    print("\n=== 1. ADDRESSING DYNAMICS — is seed 4 different? ===")
    print(f"{'seed':>5}{'cum angle':>11}{'|Δw|/|w|':>10}{'flip 500-1000':>15}"
          f"{'flip 9.5-10k':>14}{'ratio':>8}{'total flips':>13}")
    for s in SEEDS:
        r = R[s]
        mark = " <-- outlier" if s == STAR else ""
        print(f"{s:>5}{r['cum_angle']:>10.2f}°{r['cum_rel_dw']:>10.3f}"
              f"{100*r['first_flip']:>14.2f}%{100*r['last_flip']:>13.2f}%"
              f"{r['flip_ratio']:>8.2f}{100*r['total_flip']:>12.1f}%{mark}")
    print("\n  per-500-iter bit-flip rate over training (%):")
    print(f"{'iter':>7}" + "".join(f"{'s'+str(s):>8}" for s in SEEDS))
    for i, it in enumerate(R[0]["seg_it"]):
        if it % 1000 == 0:
            print(f"{it:>7}" + "".join(f"{100*R[s]['seg_flip'][i]:>8.2f}"
                                       for s in SEEDS))

    print("\n=== 2. FINAL ROW USAGE per seed (of 64 per table, 32 tables) ===")
    print(f"{'seed':>5}{'score':>9}{'mean':>8}{'p25':>7}{'min':>6}{'max':>6}"
          f"{'<32 rows':>10}{'<20 rows':>10}{'redundancy':>12}")
    for s in SEEDS:
        r = R[s]
        print(f"{s:>5}{r['score']:>9.0f}{r['rows_used_mean']:>8.1f}"
              f"{r['rows_used_p25']:>7.1f}{r['rows_used_min']:>6}{r['rows_used_max']:>6}"
              f"{r['tables_under_32']:>10}{r['tables_under_20']:>10}"
              f"{r['final_redundancy']:>12.3f}")

    print("\n=== 3. INIT DIFFERENCES (anchor_pairs: same structure, different draw) ===")
    print(f"{'seed':>5}{'init rows':>11}{'init min':>10}{'redundancy':>12}"
          f"{'dup pairs':>11}{'dims unused':>13}{'dim imbalance':>15}"
          f"{'table std':>11}")
    for s in SEEDS:
        r = R[s]
        ts = f"{r['table_init_std']:.5f}" if r["table_init_std"] else "n/a"
        print(f"{s:>5}{r['init_rows_mean']:>11.1f}{r['init_rows_min']:>10}"
              f"{r['init_redundancy']:>12.3f}{r['init_dup_pairs']:>11}"
              f"{r['init_dims_unused']:>13}{r['init_dim_gini']:>15.3f}{ts:>11}")

    # What the seed actually changes about the init. NOT the SET of pairs: this init is
    # CANONICAL_FULL_COVERAGE, which by construction uses all C(17,2) = 136 distinct
    # comparators in every seed's draw (verified below), so a Jaccard on the pair set is
    # 1.000 for every pair of seeds and measures nothing. What the seed permutes is the
    # ASSIGNMENT -- which 6 comparators are grouped into one table, and which 56 of the
    # 136 get duplicated. Grouping is what decides whether a table's 6 bits are
    # complementary or redundant, so that is the thing to measure.
    sets = {s: set(tuple(sorted(p)) for p in R[s]["pairs"]) for s in SEEDS}
    n_pairs = len(R[0]["pairs"])
    print(f"\n  every seed uses all {len(sets[0])} distinct comparators in "
          f"{n_pairs} slots (full coverage, so the pair SET is seed-independent):")
    print("    distinct pairs per seed: "
          + "  ".join(f"s{s} {len(sets[s])}" for s in SEEDS))
    print("\n  assignment overlap — fraction of the 192 (table, bit) slots holding the "
          "same comparator:")
    print("       " + "".join(f"{'s'+str(s):>8}" for s in SEEDS))
    ov = {}
    for s in SEEDS:
        row = []
        for t in SEEDS:
            if s == t:
                row.append(1.0); continue
            f = float(np.mean([tuple(sorted(a_)) == tuple(sorted(b_))
                               for a_, b_ in zip(R[s]["pairs"], R[t]["pairs"])]))
            row.append(f)
            ov.setdefault(s, []).append(f)
        print(f"    s{s} " + "".join(f"{v:>8.3f}" for v in row))
    print("    mean slot agreement with the other five: "
          + "  ".join(f"s{s} {np.mean(ov[s]):.3f}" for s in SEEDS))
    print("\n  per-table grouping overlap (mean Jaccard of the 6-comparator sets, best "
          "table-to-table match):")
    grp = {}
    for s in SEEDS:
        tabs_s = [set(tuple(sorted(p)) for p in R[s]["pairs"][t * nap:(t + 1) * nap])
                  for t in range(len(R[s]["pairs"]) // nap)]
        vals = []
        for t2 in SEEDS:
            if t2 == s:
                continue
            tabs_t = [set(tuple(sorted(p)) for p in R[t2]["pairs"][t * nap:(t + 1) * nap])
                      for t in range(len(R[t2]["pairs"]) // nap)]
            best = [max(len(a_ & b_) / len(a_ | b_) for b_ in tabs_t) for a_ in tabs_s]
            vals.append(float(np.mean(best)))
        grp[s] = float(np.mean(vals))
    print("    " + "  ".join(f"s{s} {grp[s]:.3f}" for s in SEEDS))

    print("\n=== 4. CORRELATION of score with init / dynamics properties ===")
    feats = dict(
        init_rows_mean="rows addressed at init",
        init_redundancy="bit redundancy at init",
        init_dup_pairs="duplicate pairs at init",
        init_dim_reuse="dim reuse per table at init",
        init_dims_unused="obs dims unused at init",
        rows_used_mean="rows addressed at 10k",
        final_redundancy="bit redundancy at 10k",
        cum_angle="cumulative rotation",
        total_flip="total bit churn",
        last_flip="still-moving rate at 10k",
        alpha_final="final alpha (entropy temp)",
        alpha_min="min alpha",
        iters_to_2000="iters to MJX 2000",
        iters_to_4000="iters to MJX 4000",
        table_init_std="table init std",
    )
    y = np.array([R[s]["score"] for s in SEEDS], np.float64)
    rank = lambda a: np.argsort(np.argsort(a)).astype(np.float64)
    corrs = {}
    print(f"{'property':<30}{'pearson':>9}{'spearman':>10}   values (s0..s5)")
    for k, lab in feats.items():
        v = np.array([(R[s][k] if R[s][k] is not None else np.nan) for s in SEEDS],
                     np.float64)
        if np.isnan(v).any():
            print(f"{lab:<30}{'n/a':>9}{'':>10}   not logged")
            continue
        if v.std() < 1e-12:
            print(f"{lab:<30}{'constant':>9}{'':>10}   {v[0]:.4g}")
            continue
        pr = float(np.corrcoef(v, y)[0, 1]); sr = float(np.corrcoef(rank(v), rank(y))[0, 1])
        corrs[k] = dict(pearson=pr, spearman=sr, values=v.tolist())
        print(f"{lab:<30}{pr:>9.3f}{sr:>10.3f}   "
              + " ".join(f"{t:.3g}" for t in v))
    print("\n  n = 6: |r| >= 0.81 for 5% significance. Nothing below that is a finding.")
    print("  NOTE: critic loss is NOT logged by lut_sac.py (only mjx_return, "
          "row_coverage, alpha). It cannot be recovered without retraining.")

    # ---------------- figure -------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cols = {s: ("#d62728" if s == STAR else "#9aa0a6") for s in SEEDS}
    for s in SEEDS:
        r = R[s]
        lw = 2.4 if s == STAR else 1.3
        ax[0, 0].plot(r["h_it"], r["ret"], color=cols[s], lw=lw,
                      label=f"seed {s} ({r['score']:.0f})", zorder=3 if s == STAR else 2)
        ax[0, 1].plot(r["h_it"], r["alpha"], color=cols[s], lw=lw)
        ax[1, 0].plot(r["seg_it"], 100 * np.array(r["seg_flip"]), color=cols[s], lw=lw)
    ax[0, 0].set(title="training curve (20-ep MJX proxy)", xlabel="iteration",
                 ylabel="MJX return")
    ax[0, 0].legend(fontsize=8, loc="upper left")
    ax[0, 1].set(title="entropy temperature alpha", xlabel="iteration", ylabel="alpha")
    ax[1, 0].set(title="bit-flip rate per 500 iters (addressing churn)",
                 xlabel="iteration", ylabel="% of address bits flipped")
    for s in SEEDS:
        r = R[s]
        ax[1, 1].scatter(r["rows_used_mean"], r["score"], s=90, color=cols[s], zorder=3)
        ax[1, 1].annotate(f"s{s}", (r["rows_used_mean"], r["score"]),
                          textcoords="offset points", xytext=(7, -3), fontsize=9)
    ax[1, 1].set(title="score vs rows actually addressed at 10k",
                 xlabel="mean rows used per table (of 64)", ylabel="CPU-ref return")
    for a_ in ax.ravel():
        a_.grid(alpha=0.25)
    fig.suptitle("exp_c18 — why does seed 4 (red) reach 5287 while the others sit at "
                 "4112 ± 160?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG, dpi=125)
    print(f"\nwrote {FIG}")

    json.dump(dict(seeds={str(s): {k: v for k, v in R[s].items() if k != "pairs"}
                          for s in SEEDS},
                   correlations=corrs,
                   pair_overlap={str(s): float(np.mean(ov[s])) for s in SEEDS},
                   n_obs=int(x.shape[0]),
                   critic_loss_available=False),
              open(os.path.join(HERE, "seed4_diagnostics.json"), "w"), indent=1)
    print("wrote seed4_diagnostics.json")


if __name__ == "__main__":
    main()
