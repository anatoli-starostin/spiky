"""exp_c08 — does choosing SAC as the teacher buy robustness in the cloned LUT? (#75)

Overlays the SAC-taught LUT on the exp_c07 grid and correlates its degradation profile
against its own teacher, the PPO-taught LUT, and the PPO teacher.
"""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C07 = os.path.join(HERE, "..", "exp_c07_robustness")

rows = []
for p in (os.path.join(C07, "results_torch.json"), os.path.join(C07, "results_jax.json"),
          os.path.join(HERE, "results_sac_lut.json")):
    if os.path.exists(p):
        rows += json.load(open(p))

ORDER = ["SAC-MLP", "LUT-SAC-distilled", "PPO-MLP", "LUT-distilled"]
NICE = {"SAC-MLP": "SAC-MLP (teacher)", "LUT-SAC-distilled": "LUT ← SAC (5k)",
        "PPO-MLP": "PPO-MLP (teacher)", "LUT-distilled": "LUT ← PPO (5k)"}
COLOR = {"SAC-MLP": "#6b7785", "LUT-SAC-distilled": "#7a5ea7",
         "PPO-MLP": "#2f6f4f", "LUT-distilled": "#3b6ea5"}
STYLE = {"SAC-MLP": "-", "LUT-SAC-distilled": "--", "PPO-MLP": "-",
         "LUT-distilled": "--"}
AXES = ["mass", "gravity", "friction", "geometry"]
LABEL = {"mass": "body-mass scale", "gravity": "gravity scale",
         "friction": "ground-friction scale", "geometry": "limb-geometry scale"}
INK = "#1d2733"; MUTED = "#6b7785"; GRID = "#dfe3e8"; BAR = "#b4553a"


def get(p, ax):
    r = sorted([x for x in rows if x["policy"] == p and x["axis"] == ax],
               key=lambda x: x["value"])
    return [x["value"] for x in r], [x["mean"] for x in r], [x["std"] for x in r]


def profile(p):
    r = sorted([x for x in rows if x["policy"] == p],
               key=lambda x: (x["axis"], x["value"]))
    return np.array([x["mean"] for x in r])


def nominal(p):
    v = [x["mean"] for x in rows if x["policy"] == p and x["value"] == 1.0]
    return float(np.mean(v))


print("=== nominal ===")
for p in ORDER:
    print(f"  {NICE[p]:<22}{nominal(p):8.1f}")

print("\n=== worst-case retained fraction of own nominal ===")
print(f"{'policy':<22}" + "".join(f"{a:>10}" for a in AXES) + f"{'mean':>9}")
ret = {}
for p in ORDER:
    base = nominal(p)
    w = []
    for ax in AXES:
        vs, ms, _ = get(p, ax)
        off = [m / base for v, m in zip(vs, ms) if v != 1.0]
        w.append(min(off) if off else 1.0)
    ret[p] = w
    print(f"{NICE[p]:<22}" + "".join(f"{x*100:9.1f}%" for x in w)
          + f"{np.mean(w)*100:8.1f}%")

print("\n=== cells clearing the 3000 bar (of 18) ===")
for p in ORDER:
    n = sum(1 for x in rows if x["policy"] == p and x["mean"] >= 3000)
    print(f"  {NICE[p]:<22}{n:>3}/18")

print("\n=== correlation of the 18-point degradation profile ===")
P = {p: profile(p) for p in ORDER}
print(f"{'':<22}" + "".join(f"{NICE[p][:14]:>16}" for p in ORDER))
for a_ in ORDER:
    print(f"{NICE[a_]:<22}" + "".join(
        f"{np.corrcoef(P[a_], P[b])[0, 1]:>16.3f}" for b in ORDER))

fig, axs = plt.subplots(1, 4, figsize=(17.5, 4.4), sharey=True)
for k, ax in enumerate(AXES):
    a_ = axs[k]
    for p in ORDER:
        vs, ms, sd = get(p, ax)
        if not vs:
            continue
        a_.errorbar(vs, ms, yerr=sd, fmt="o" + STYLE[p], color=COLOR[p], lw=1.9,
                    ms=5, capsize=2.5, label=NICE[p] if k == 0 else None)
    a_.axhline(3000, color=BAR, ls="-.", lw=1.4)
    a_.axvline(1.0, color=GRID, lw=1.4)
    a_.set_xlabel(LABEL[ax]); a_.set_title(ax, fontsize=11, color=INK)
    a_.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a_.spines[s].set_color(GRID)
    a_.tick_params(colors=MUTED, labelsize=9)
    a_.xaxis.label.set_color(MUTED)
axs[0].set_ylabel("return — CPU reference, 100 deterministic episodes")
axs[0].yaxis.label.set_color(MUTED)
axs[0].legend(fontsize=8.5, loc="lower center", frameon=False)
fig.suptitle("Does the teacher's robustness transfer to the cloned LUT? "
             "(solid = teacher MLP, dashed = its 5,378-param LUT student)",
             fontsize=12, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, "sac_vs_ppo_taught_lut.png")
fig.savefig(out, dpi=150, facecolor="white")
print("\nwrote", out)

json.dump(dict(nominal={p: nominal(p) for p in ORDER},
               retained={p: ret[p] for p in ORDER},
               corr={a_: {b: float(np.corrcoef(P[a_], P[b])[0, 1]) for b in ORDER}
                     for a_ in ORDER}),
          open(os.path.join(HERE, "compare_summary.json"), "w"), indent=1)
