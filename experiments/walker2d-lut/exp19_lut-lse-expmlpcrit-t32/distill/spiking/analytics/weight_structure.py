"""(3) Weight / synapse structure: what does the surviving connectome actually look like?

Weight distributions (exc vs inh), how much of the net is DEAD (near-zero weight, doing no
work) versus effective, exc/inh balance, and structural drift over rounds taken from the
history's n_syn / n_exc / n_inh (the checkpoint is overwritten each round, so per-round genome
state is not recoverable).

Reads history + final checkpoint. No GPU.

    python weight_structure.py [experiment ...]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import INH, OUT, available, load_genomes, load_history

W_MAX = 30.0
CEIL = 1.5 * W_MAX          # the meta's max_weight, where STDP clips
DEAD = 0.02 * W_MAX         # the prune threshold used by mutate_structural

want = sys.argv[1:] or None
exps = {k: v for k, v in available().items() if not want or k in want}
withck = {k: v for k, v in exps.items() if v["ckpt"]}

fig, axes = plt.subplots(2, max(1, len(withck)), figsize=(5.2 * max(1, len(withck)), 8),
                         squeeze=False)
summary = {}
for j, (name, meta) in enumerate(withck.items()):
    genomes, ewma, _, _, nxt = load_genomes(meta["ckpt"])
    we = np.concatenate([g["weight"][g["src_pool"] != INH] for g in genomes])
    wi = np.concatenate([g["weight"][g["src_pool"] == INH] for g in genomes])

    ax = axes[0, j]
    ax.hist(we, bins=60, color="#4E79A7")
    ax.axvline(DEAD, color="k", ls=":", lw=1, label=f"dead threshold {DEAD:g}")
    ax.axvline(CEIL, color="r", ls=":", lw=1, label=f"clip ceiling {CEIL:g}")
    ax.set_yscale("log")
    ax.set_title(f"{name}: excitatory weights (round {nxt})")
    ax.set_xlabel("weight"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

    ax = axes[1, j]
    h = load_history(meta["history"])
    r = [x["rnd"] for x in h]
    k = meta["k"]
    ax.plot(r, [x["n_exc"] / k for x in h], label="exc/net", color="#4E79A7")
    ax.plot(r, [x["n_inh"] / k for x in h], label="inh/net", color="#E15759")
    ax.set_title(f"{name}: synapse counts per net")
    ax.set_xlabel("round"); ax.grid(alpha=0.25); ax.legend(fontsize=8)

    dead = int((np.abs(we) < DEAD).sum())
    clipped = int((we >= CEIL - 1e-3).sum())
    summary[name] = dict(
        round=nxt, n_exc=int(we.size), n_inh=int(wi.size),
        exc_inh_ratio=float(we.size / max(wi.size, 1)),
        exc_mean=float(we.mean()), exc_median=float(np.median(we)),
        exc_std=float(we.std()), exc_max=float(we.max()),
        dead_frac=float(dead / max(we.size, 1)),
        clipped_frac=float(clipped / max(we.size, 1)),
        inh_all_pinned=bool(np.allclose(wi, -5.0)),
        exc_first=int(h[0]["n_exc"] / k), exc_last=int(h[-1]["n_exc"] / k),
        inh_first=int(h[0]["n_inh"] / k), inh_last=int(h[-1]["n_inh"] / k))
    s = summary[name]
    print(f"{name} (round {nxt}):")
    print(f"   exc {s['n_exc']:,} synapses  mean {s['exc_mean']:.3f}  median "
          f"{s['exc_median']:.3f}  std {s['exc_std']:.3f}  max {s['exc_max']:.2f}")
    print(f"   DEAD (|w| < {DEAD:g}): {s['dead_frac']:.1%}   "
          f"AT CLIP CEILING ({CEIL:g}): {s['clipped_frac']:.1%}")
    print(f"   inh {s['n_inh']:,} synapses, all pinned at -5: {s['inh_all_pinned']}   "
          f"exc/inh ratio {s['exc_inh_ratio']:.2f}")
    print(f"   drift per net: exc {s['exc_first']:,} -> {s['exc_last']:,}   "
          f"inh {s['inh_first']:,} -> {s['inh_last']:,}")

fig.tight_layout()
p = f"{OUT}/weight_structure.png"
fig.savefig(p, dpi=130)
json.dump(summary, open(f"{OUT}/weight_structure.json", "w"), indent=1)
print(f"\nwrote {p} and {OUT}/weight_structure.json")
