"""(2) Delay-code utilisation: which excitatory delays do surviving synapses actually use?

If the population only ever uses a narrow slice of [1, D_MAX], widening the range to 1-48 buys
nothing; if it saturates the top of the range, widening should help. Both an unweighted count
and a |weight|-weighted count are reported, because a delay used by many near-zero synapses is
not really "used".

Reads the final checkpoint only. No GPU.

    python delay_utilization.py            # every experiment with a checkpoint
    python delay_utilization.py k128
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import INH, OUT, available, load_genomes

want = sys.argv[1:] or None
exps = {k: v for k, v in available().items()
        if v["ckpt"] and (not want or k in want)}
if not exps:
    print("no experiments with checkpoints yet")
    sys.exit(0)

fig, axes = plt.subplots(len(exps), 1, figsize=(11, 3.4 * len(exps)), squeeze=False)
summary = {}
for ax, (name, meta) in zip(axes[:, 0], exps.items()):
    genomes, ewma, _, _, nxt = load_genomes(meta["ckpt"])
    d_max = meta["d_max"]
    dl = np.concatenate([g["delay"][g["src_pool"] != INH] for g in genomes])
    w = np.abs(np.concatenate([g["weight"][g["src_pool"] != INH] for g in genomes]))
    bins = np.arange(1, d_max + 2)
    cnt, _ = np.histogram(dl, bins=bins)
    wsum, _ = np.histogram(dl, bins=bins, weights=w)
    # "effective" = carrying non-negligible weight (2% of w_max=30 is the prune threshold)
    eff = np.histogram(dl[w >= 0.6], bins=bins)[0]

    x = bins[:-1]
    ax.bar(x - 0.2, cnt / cnt.sum(), width=0.4, label="share of synapses", color="#4E79A7")
    ax.bar(x + 0.2, wsum / wsum.sum(), width=0.4, label="share of |weight|", color="#E15759")
    ax.set_title(f"{name}: excitatory delay utilisation over [1,{d_max}] "
                 f"({dl.size:,} exc synapses at round {nxt})")
    ax.set_xlabel("delay (ticks)")
    ax.set_ylabel("share")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    share = cnt / cnt.sum()
    top_q = int(np.ceil(d_max * 0.75))
    summary[name] = dict(
        d_max=d_max, n_exc=int(dl.size), mean_delay=float(dl.mean()),
        median_delay=float(np.median(dl)),
        share_top_quarter=float(share[top_q - 1:].sum()),
        share_bottom_quarter=float(share[:max(1, d_max // 4)].sum()),
        max_share_delay=int(x[share.argmax()]), max_share=float(share.max()),
        min_share=float(share.min()), unused_delays=int((cnt == 0).sum()),
        effective_frac=float(eff.sum() / max(cnt.sum(), 1)),
        weighted_mean_delay=float((wsum * x).sum() / max(wsum.sum(), 1e-9)))
    s = summary[name]
    print(f"{name}: {dl.size:,} exc synapses, delays [1,{d_max}]")
    print(f"   mean delay {s['mean_delay']:.2f}  median {s['median_delay']:.0f}  "
          f"|w|-weighted mean {s['weighted_mean_delay']:.2f}")
    print(f"   share in TOP quarter of range {s['share_top_quarter']:.1%}   "
          f"bottom quarter {s['share_bottom_quarter']:.1%}")
    print(f"   most-used delay {s['max_share_delay']} ({s['max_share']:.1%}), "
          f"least-used share {s['min_share']:.1%}, unused delays {s['unused_delays']}")
    print(f"   carrying non-negligible weight (|w| >= 0.6): {s['effective_frac']:.1%}")

fig.tight_layout()
p = f"{OUT}/delay_utilization.png"
fig.savefig(p, dpi=130)
json.dump(summary, open(f"{OUT}/delay_utilization.json", "w"), indent=1)
print(f"\nwrote {p} and {OUT}/delay_utilization.json")
