"""(4) Teacher-vs-student error breakdown on the held-out set. NEEDS GPU -- queue it.

The distillation dataset already carries the teacher's output: load() returns (X, Y) where Y
is the FastMultiHeadLUT teacher's action vector for each state. So no separate teacher forward
pass is needed -- the comparison is the student's first-spike code against Y.

Reports, for the best member of a finished experiment:
  * per-action-dim agreement (Walker2d has 6 dims): pairwise-order agreement and Spearman-like
    rank correlation of -first_spike against the teacher action, dimension by dimension
  * error vs STATE REGION, binned by teacher action magnitude, so we can see whether the
    student fails on large/aggressive actions or on the quiet ones
  * silence rate per dim (an output that never spikes carries no rank information)

    python teacher_student.py k128 [small8020 ...]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "..")
from common import OUT, available, load_genomes            # noqa: E402
import steady_state as S                                    # noqa: E402


def student_first_spikes(genome, Xv, enc, d_max):
    """Build the single best net and return its first-spike matrix [B, 6]."""
    S.D_MAX = d_max
    S.N_DELAY_METAS = S.D_MAX - S.D_MIN + 1
    h = S.build_pool([genome], "cuda", seed=1, stdp_lr=0.01, w_max=30.0)
    first, _ = S.run_episode(h, Xv, enc, 200.0, train=False)
    torch.cuda.synchronize()
    out = first[:, 0, :]            # [B, N_OUT]
    del h
    torch.cuda.empty_cache()
    return out


def analyse(name, meta, n_val=2000):
    genomes, ewma, _, _, nxt = load_genomes(meta["ckpt"])
    best_i = int(np.nanargmax(ewma))
    X, Y, Xpool, Ypool, Xval, Yval = S.load(64, 0, n_val)
    enc = S.LatencyEncoder(Xpool)
    first = student_first_spikes(genomes[best_i], Xval, enc, meta["d_max"])
    pred = -first                                    # earlier spike == larger action
    n_out = pred.shape[1]

    # per-dimension: does the student rank this dim vs the others the way the teacher does?
    iu, ju = np.triu_indices(n_out, 1)
    dp = np.sign(pred[:, iu] - pred[:, ju])
    dt = np.sign(Yval[:, iu] - Yval[:, ju])
    per_pair = (dp == dt).mean(0)
    per_dim = np.zeros(n_out)
    for d in range(n_out):
        m = (iu == d) | (ju == d)
        per_dim[d] = per_pair[m].mean()
    silence = (first >= S.N_TICKS).mean(0)

    # error vs region, binned by teacher action magnitude
    mag = np.abs(Yval).mean(1)
    edges = np.quantile(mag, np.linspace(0, 1, 6))
    agree_by_bin, centres = [], []
    for b in range(5):
        m = (mag >= edges[b]) & (mag <= edges[b + 1])
        if m.sum() < 10:
            continue
        d2p = np.sign(pred[m][:, iu] - pred[m][:, ju])
        d2t = np.sign(Yval[m][:, iu] - Yval[m][:, ju])
        agree_by_bin.append(float((d2p == d2t).mean()))
        centres.append(float(mag[m].mean()))

    res = dict(best_member=best_i, round=nxt, n_val=int(Xval.shape[0]),
               overall_pair_agreement=float((dp == dt).mean()),
               per_dim_agreement=[float(v) for v in per_dim],
               silence_rate_per_dim=[float(v) for v in silence],
               agreement_by_action_magnitude=agree_by_bin,
               magnitude_bin_centres=centres)
    print(f"{name}: best member {best_i} at round {nxt}, {res['n_val']} held-out states")
    print(f"   overall pairwise-order agreement with teacher: "
          f"{res['overall_pair_agreement']:.3f}  (0.5 = chance)")
    print("   per action dim: " + "  ".join(f"d{d}={v:.3f}" for d, v in enumerate(per_dim)))
    print("   silence rate  : " + "  ".join(f"d{d}={v:.2f}" for d, v in enumerate(silence)))
    print("   by |action| bin: " + "  ".join(f"{c:.2f}->{a:.3f}"
                                             for c, a in zip(centres, agree_by_bin)))
    return res


if __name__ == "__main__":
    want = sys.argv[1:] or None
    exps = {k: v for k, v in available().items()
            if v["ckpt"] and (not want or k in want)}
    summary = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, meta in exps.items():
        try:
            r = analyse(name, meta)
            summary[name] = r
            ax[0].plot(range(len(r["per_dim_agreement"])), r["per_dim_agreement"],
                       marker="o", label=name)
            ax[1].plot(r["magnitude_bin_centres"], r["agreement_by_action_magnitude"],
                       marker="o", label=name)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:120]}")
    ax[0].set_xlabel("action dim"); ax[0].set_ylabel("pairwise-order agreement")
    ax[0].set_title("Student vs teacher, per action dim"); ax[0].axhline(0.5, color="k", ls=":")
    ax[1].set_xlabel("mean |teacher action| (state region)")
    ax[1].set_title("Agreement vs action magnitude"); ax[1].axhline(0.5, color="k", ls=":")
    for a in ax:
        a.grid(alpha=0.25); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/teacher_student.png", dpi=130)
    json.dump(summary, open(f"{OUT}/teacher_student.json", "w"), indent=1)
    print(f"\nwrote {OUT}/teacher_student.png and .json")
