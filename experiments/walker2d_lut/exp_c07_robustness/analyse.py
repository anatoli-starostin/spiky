"""exp_c07 — degradation curves + the robustness verdict (#75)."""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWN = {"LUT-distilled": 5511.9, "SAC-MLP": 5273.4,
         "PPO-MLP": 5555.5, "LUT-scratch": 4406.9}
PARAMS = {"LUT-distilled": 5378, "SAC-MLP": 73484,
          "PPO-MLP": 71948, "LUT-scratch": 26880}
ORDER = ["PPO-MLP", "SAC-MLP", "LUT-distilled", "LUT-scratch"]
COLOR = {"PPO-MLP": "#2f6f4f", "SAC-MLP": "#6b7785",
         "LUT-distilled": "#3b6ea5", "LUT-scratch": "#c98a2b"}
STYLE = {"PPO-MLP": "-", "SAC-MLP": "-", "LUT-distilled": "--", "LUT-scratch": "--"}
AXES = ["mass", "gravity", "friction", "geometry"]
LABEL = {"mass": "body-mass scale", "gravity": "gravity scale",
         "friction": "ground-friction scale", "geometry": "limb-geometry scale"}
INK = "#1d2733"; MUTED = "#6b7785"; GRID = "#dfe3e8"; BAR = "#b4553a"

rows = []
for f in ("results_torch.json", "results_jax.json"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        rows += json.load(open(p))


def get(pol, axis):
    r = sorted([x for x in rows if x["policy"] == pol and x["axis"] == axis],
               key=lambda x: x["value"])
    return [x["value"] for x in r], [x["mean"] for x in r], [x["std"] for x in r]


def nominal(pol):
    v = [x["mean"] for x in rows if x["policy"] == pol and x["value"] == 1.0]
    return float(np.mean(v)) if v else float("nan")


print("=== nominal sanity check (value = 1.0 must reproduce the known score) ===")
print(f"{'policy':<16}{'params':>9}{'harness nominal':>18}{'known':>10}{'Δ':>8}")
sanity = {}
for p in ORDER:
    n = nominal(p)
    sanity[p] = n
    print(f"{p:<16}{PARAMS[p]:>9,}{n:>18.1f}{KNOWN[p]:>10.1f}"
          f"{100*(n-KNOWN[p])/KNOWN[p]:>7.1f}%")

print("\n=== retained fraction of each policy's OWN nominal ===")
summary = {}
for p in ORDER:
    base = sanity[p]
    worst, breaks = [], []
    for ax in AXES:
        vs, ms, _ = get(p, ax)
        frac = [m / base for m in ms]
        off = [f for v, f in zip(vs, frac) if v != 1.0]
        worst.append(min(off) if off else 1.0)
        # how far from nominal can we go before dropping below 3000?
        ok = [v for v, m in zip(vs, ms) if m >= 3000]
        breaks.append((min(ok), max(ok)) if ok else (None, None))
    summary[p] = dict(worst=worst, breaks=breaks,
                      mean_retained=float(np.mean(worst)))
    print(f"{p:<16} worst-case retained per axis: " +
          "  ".join(f"{ax[:4]} {w*100:5.1f}%" for ax, w in zip(AXES, worst)) +
          f"   | mean {np.mean(worst)*100:5.1f}%")

print("\n=== range still clearing the 3000 bar (per axis) ===")
for p in ORDER:
    s = " ".join(f"{ax[:4]}[{b[0] if b[0] is not None else '-'}"
                 f"..{b[1] if b[1] is not None else '-'}]"
                 for ax, b in zip(AXES, summary[p]["breaks"]))
    print(f"{p:<16} {s}")

fig, axs = plt.subplots(1, 4, figsize=(17.5, 4.4), sharey=True)
for k, ax in enumerate(AXES):
    a = axs[k]
    for p in ORDER:
        vs, ms, sd = get(p, ax)
        if not vs:
            continue
        a.errorbar(vs, ms, yerr=sd, fmt="o" + STYLE[p], color=COLOR[p], lw=1.9,
                   ms=5, capsize=2.5,
                   label=f"{p} ({PARAMS[p]/1000:.0f}k)" if k == 0 else None)
    a.axhline(3000, color=BAR, ls="-.", lw=1.4)
    if k == 0:
        a.text(min(get(ORDER[0], ax)[0]), 3150, "solved = 3000", color=BAR, fontsize=8.5)
    a.axvline(1.0, color=GRID, lw=1.4)
    a.set_xlabel(LABEL[ax])
    a.set_title(ax, fontsize=11, color=INK)
    a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GRID)
    a.tick_params(colors=MUTED, labelsize=9)
    a.xaxis.label.set_color(MUTED)
axs[0].set_ylabel("return — CPU reference, 100 deterministic episodes")
axs[0].yaxis.label.set_color(MUTED)
axs[0].legend(fontsize=8.5, loc="lower center", frameon=False)
fig.suptitle("Zero-shot robustness: frozen policies under perturbed Walker2d dynamics "
             "(solid = MLP, dashed = LUT)", fontsize=12, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, "robustness_curves.png")
fig.savefig(out, dpi=150, facecolor="white")
print("\nwrote", out)

json.dump(dict(nominal=sanity, known=KNOWN, params=PARAMS,
               summary={k: {"worst_retained_per_axis": v["worst"],
                            "mean_retained": v["mean_retained"]}
                        for k, v in summary.items()}, rows=rows),
          open(os.path.join(HERE, "robustness_summary.json"), "w"), indent=1)
