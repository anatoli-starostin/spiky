"""Does a net's TIE RATE explain high tau-B fitness alongside low teacher agreement?

The two metrics treat ties oppositely:
  * tau-B (the FITNESS) removes ties from its denominator -- den = sqrt((n0-n1)*(n0-n2)) --
    so a net that ties a lot is graded only on the pairs it does order strictly.
  * teacher agreement counts a student tie against a strict teacher order as an ERROR.
Student ties are possible at all because first-spike is an INTEGER tick in [0,96]: two outputs
firing on the same tick tie exactly. So the prediction is: higher tie rate -> higher tau-B and
lower agreement, simultaneously.

Quiet by design: everything goes to a log file, no progress bars on any shared console.

    python tie_rate.py                    # every experiment with a checkpoint
    python tie_rate.py k128 delay148
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "..")
from common import OUT, available, load_genomes            # noqa: E402
import steady_state as S                                    # noqa: E402
from harness import kendall_tau_b                        # noqa: E402


def rollout(genome, Xv, enc, d_max):
    import torch
    S.D_MAX = d_max
    S.N_DELAY_METAS = S.D_MAX - S.D_MIN + 1
    h = S.build_pool([genome], "cuda", seed=1, stdp_lr=0.01, w_max=30.0)
    first, _ = S.run_episode(h, Xv, enc, 200.0, train=False)
    torch.cuda.synchronize()
    out = first[:, 0, :]
    del h
    torch.cuda.empty_cache()
    return out


def analyse(name, meta, n_val=2000):
    genomes, ewma, _, _, nxt = load_genomes(meta["ckpt"])
    best_i = int(np.nanargmax(ewma))
    X, Y, Xpool, Ypool, Xval, Yval = S.load(64, 0, n_val)
    enc = S.LatencyEncoder(Xpool)
    first = rollout(genomes[best_i], Xval, enc, meta["d_max"])
    pred = -first
    n_out = pred.shape[1]

    iu, ju = np.triu_indices(n_out, 1)
    dp = np.sign(pred[:, iu] - pred[:, ju])
    dt = np.sign(Yval[:, iu] - Yval[:, ju])

    tie_rate = float((dp == 0).mean())
    teacher_tie_rate = float((dt == 0).mean())
    agreement = float((dp == dt).mean())
    # agreement restricted to the pairs the student actually ORDERS -- removes the tie penalty
    nz = dp != 0
    agreement_untied = float((dp[nz] == dt[nz]).mean()) if nz.any() else float("nan")

    # tau-B on the same data: raw, and corrected by this net's own label-shuffle null
    raw_tau = float(kendall_tau_b(pred, Yval).mean())
    rng = np.random.default_rng(0)
    null = float(np.mean([kendall_tau_b(pred, Yval[rng.permutation(Yval.shape[0])]).mean()
                          for _ in range(20)]))
    corrected_tau = raw_tau - null

    # per-dim tie involvement, and how concentrated the first-spike ticks are
    tie_cells = (dp == 0)
    per_dim_tie = [float(tie_cells[:, (iu == d) | (ju == d)].mean()) for d in range(n_out)]
    ticks = first.astype(int)
    distinct_per_state = float(np.mean([len(np.unique(r)) for r in ticks]))
    return dict(best_member=best_i, round=nxt,
                tie_rate=tie_rate, teacher_tie_rate=teacher_tie_rate,
                agreement=agreement, agreement_untied=agreement_untied,
                raw_tau=raw_tau, null=null, corrected_tau=corrected_tau,
                per_dim_tie=per_dim_tie,
                distinct_ticks_per_state=distinct_per_state,
                distinct_ticks_overall=int(np.unique(ticks).size),
                tick_min=int(ticks.min()), tick_max=int(ticks.max()),
                tick_mean=float(ticks.mean()), tick_std=float(ticks.std()))


if __name__ == "__main__":
    want = sys.argv[1:] or None
    exps = {k: v for k, v in available().items()
            if v["ckpt"] and (not want or k in want)}
    res = {}
    for name, meta in exps.items():
        try:
            res[name] = analyse(name, meta)
            print(f"done {name}", flush=True)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)

    hdr = (f"{'experiment':12s} {'tie rate':>9s} {'agree':>7s} {'agree|untied':>13s} "
           f"{'raw tau':>8s} {'corr tau':>9s} {'distinct ticks/state':>21s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for n, r in res.items():
        print(f"{n:12s} {r['tie_rate']:9.3f} {r['agreement']:7.3f} "
              f"{r['agreement_untied']:13.3f} {r['raw_tau']:8.3f} "
              f"{r['corrected_tau']:9.3f} {r['distinct_ticks_per_state']:21.2f}")
    for n, r in res.items():
        print(f"\n{n}: ticks [{r['tick_min']}, {r['tick_max']}] mean {r['tick_mean']:.1f} "
              f"std {r['tick_std']:.1f}, {r['distinct_ticks_overall']} distinct values overall")
        print("   per-dim tie involvement: " +
              "  ".join(f"d{d}={v:.3f}" for d, v in enumerate(r['per_dim_tie'])))

    if res:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        names = list(res)
        tr = [res[n]["tie_rate"] for n in names]
        ax[0].scatter(tr, [res[n]["corrected_tau"] for n in names])
        ax[1].scatter(tr, [res[n]["agreement"] for n in names])
        for i, n in enumerate(names):
            ax[0].annotate(n, (tr[i], res[n]["corrected_tau"]), fontsize=8)
            ax[1].annotate(n, (tr[i], res[n]["agreement"]), fontsize=8)
        ax[0].set_xlabel("tie rate"); ax[0].set_ylabel("corrected tau-B (fitness)")
        ax[1].set_xlabel("tie rate"); ax[1].set_ylabel("teacher agreement")
        for a in ax:
            a.grid(alpha=0.25)
        fig.suptitle("Does tying inflate tau-B while depressing teacher agreement?")
        fig.tight_layout()
        fig.savefig(f"{OUT}/tie_rate.png", dpi=130)
        json.dump(res, open(f"{OUT}/tie_rate.json", "w"), indent=1)
        print(f"\nwrote {OUT}/tie_rate.png and .json")
