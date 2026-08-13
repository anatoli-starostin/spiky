"""exp_c18 — figure for the seed-4 behaviour deep-dive (#75). SPIKY venv (matplotlib).

The point of the figure is the middle panel: Walker2d-v5's return is
`steps_survived + total forward distance - 1e-3 * total control cost`, so once a policy
survives the full horizon its score IS its mean forward velocity, on a line of slope 1000.
Plotting the seeds against that line shows immediately that seed 4 is not off the line --
it is further along it.
"""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "behavior_stats.json")))
A = np.load(os.path.join(HERE, "behavior_arrays.npz"))
SEEDS = J["seeds"]
STAR = 4
FIG = os.path.join(HERE, "seed4_behavior.png")
col = {s: ("#d62728" if s == STAR else "#5b7fa6") for s in SEEDS}

fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

# --- 1. per-episode return distributions ---------------------------------
for i, s in enumerate(SEEDS):
    r = A[f"ret_{s}"]
    x = i + 0.06 * np.random.default_rng(s).standard_normal(len(r))
    ax[0].scatter(x, r, s=7, alpha=0.45, color=col[s], edgecolors="none")
    ax[0].plot([i - 0.3, i + 0.3], [r.mean()] * 2, color=col[s], lw=2.5)
ax[0].set(xticks=range(len(SEEDS)), xticklabels=[f"s{s}" for s in SEEDS],
          ylabel="return (100 deterministic episodes)",
          title="per-episode return — seed 4 is higher, not just luckier")
ax[0].grid(alpha=0.25, axis="y")

# --- 2. the decomposition: score IS velocity ------------------------------
v = np.array([J["gait"][str(s)]["vel_mean"]["mean"] for s in SEEDS])
sc = np.array([J["gait"][str(s)]["score"] for s in SEEDS])
full = np.array([J["gait"][str(s)]["n_full_horizon"] for s in SEEDS])
vv = np.linspace(v.min() - 0.15, v.max() + 0.15, 50)
ax[1].plot(vv, 1000 + 1000 * vv - 3.3, color="#999", ls="--", lw=1.4,
           label="1000 steps + 1000·velocity − control cost")
for s, vi, si, fu in zip(SEEDS, v, sc, full):
    ax[1].scatter(vi, si, s=140 if s == STAR else 95, color=col[s], zorder=3,
                  marker="o" if fu == 100 else "^")
    ax[1].annotate(f"s{s}" + ("" if fu == 100 else f" ({100-fu} falls)"),
                   (vi, si), textcoords="offset points", xytext=(8, -4), fontsize=9)
ax[1].set(xlabel="mean forward velocity (m/s)", ylabel="CPU-reference return",
          title="the whole advantage is speed\n(circles = 100/100 full horizon, "
                "triangles fall sometimes)")
ax[1].legend(fontsize=8, loc="upper left")
ax[1].grid(alpha=0.25)

# --- 3. function-space distance matrix ------------------------------------
Dm = np.array(J["distance_matrix"])
im = ax[2].imshow(Dm, cmap="viridis")
ax[2].set(xticks=range(len(SEEDS)), yticks=range(len(SEEDS)),
          xticklabels=[f"s{s}" for s in SEEDS],
          yticklabels=[f"s{s}" for s in SEEDS],
          title="RMS action difference on 20k common states\n"
                "seed 4 is no further out than anyone else")
for i in range(len(SEEDS)):
    for j in range(len(SEEDS)):
        ax[2].text(j, i, f"{Dm[i, j]:.2f}", ha="center", va="center",
                   color="w" if Dm[i, j] < 0.8 else "k", fontsize=8)
fig.colorbar(im, ax=ax[2], fraction=0.046)

fig.suptitle("exp_c18 — seed 4 (red) wins by walking FASTER, not by falling less "
             "or by being a special model", fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=125)
print(f"wrote {FIG}")

# --- the arithmetic, printed so it can be checked -------------------------
print("\nreturn decomposition check (Walker2d-v5: steps + distance - control cost):")
print(f"{'seed':>5}{'steps':>7}{'+1000*v':>10}{'-cost':>8}{'= predicted':>13}"
      f"{'actual':>9}{'err':>7}")
for s in SEEDS:
    g = J["gait"][str(s)]
    steps = g["length"]["mean"]
    pred = steps + steps * g["vel_mean"]["mean"] - 1e-3 * steps * g["act_energy"]["mean"]
    print(f"{s:>5}{steps:>7.0f}{steps*g['vel_mean']['mean']:>10.0f}"
          f"{-1e-3*steps*g['act_energy']['mean']:>8.1f}{pred:>13.1f}"
          f"{g['score']:>9.1f}{pred-g['score']:>7.1f}")
surv = [s for s in SEEDS if J["gait"][str(s)]["n_full_horizon"] == 100 and s != STAR]
vs = np.mean([J["gait"][str(s)]["vel_mean"]["mean"] for s in surv])
ss = np.mean([J["gait"][str(s)]["score"] for s in surv])
g4 = J["gait"][str(STAR)]
print(f"\nagainst the {len(surv)} pack seeds that also never fall (s{surv}):")
print(f"  velocity {g4['vel_mean']['mean']:.3f} vs {vs:.3f}  -> +"
      f"{g4['vel_mean']['mean']-vs:.3f} m/s  x 1000 steps = +"
      f"{1000*(g4['vel_mean']['mean']-vs):.0f} return")
print(f"  actual score gap: {g4['score']:.1f} - {ss:.1f} = +{g4['score']-ss:.1f}")
