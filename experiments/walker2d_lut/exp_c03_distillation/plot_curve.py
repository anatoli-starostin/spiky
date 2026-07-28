"""exp_c03 — the representability curve figure (#75)."""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TEACHER = 5555.5
SAC = 5273.4
BAR = 3000

INK = "#1d2733"; MUTED = "#6b7785"; GRID = "#dfe3e8"
C_HP = "#3b6ea5"; C_FAST = "#c98a2b"; C_T = "#2f6f4f"; C_BAR = "#b4553a"

d = json.load(open(os.path.join(HERE, "sweep_results.json")))
rows = [r for r in d["results"] if not r.get("failed")]
# fold in any reruns that landed after the sweep
for f in os.listdir(HERE):
    if f.startswith("result_") and f.endswith(".json"):
        r = json.load(open(os.path.join(HERE, f)))
        if r["name"] not in {x["name"] for x in rows}:
            rows.append(r)
hp = sorted([r for r in rows if r["module"] == "hyperplane"], key=lambda r: r["total_params"])
fa = sorted([r for r in rows if r["module"] == "fast"], key=lambda r: r["total_params"])

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.9))

a = ax[0]
a.axhline(TEACHER, color=C_T, ls="--", lw=1.6)
a.text(1.5e3, TEACHER + 130, f"PPO teacher {TEACHER:.0f}", color=C_T, fontsize=9)
a.axhline(SAC, color=MUTED, ls=":", lw=1.4)
a.text(1.5e3, SAC - 330, f"SAC baseline {SAC:.0f}", color=MUTED, fontsize=9)
a.axhline(BAR, color=C_BAR, ls="-.", lw=1.5)
a.text(1.5e3, BAR + 130, "solved = 3000", color=C_BAR, fontsize=9)
a.errorbar([r["total_params"] for r in hp], [r["eval_mean"] for r in hp],
           yerr=[r["eval_std"] for r in hp], fmt="o-", color=C_HP, lw=2, ms=6,
           capsize=3, label="HyperplaneMHL (learned addressing)")
a.errorbar([r["total_params"] for r in fa], [r["eval_mean"] for r in fa],
           yerr=[r["eval_std"] for r in fa], fmt="s--", color=C_FAST, lw=2, ms=6,
           capsize=3, label="FastMHL (fixed anchor pairs)")
sm = min([r for r in hp if r["solved"]], key=lambda r: r["total_params"])
a.annotate(f"smallest solving LUT\n{sm['total_params']:,} params → {sm['eval_mean']:.0f}",
           xy=(sm["total_params"], sm["eval_mean"]), xytext=(6e3, 2100),
           fontsize=9, color=INK,
           arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
a.set_xscale("log")
a.set_xlabel("LUT policy parameters (table + addressing)")
a.set_ylabel("return — CPU reference, 100 deterministic episodes")
a.set_title("A  Representability: a LUT policy vs its size\n"
            "distilled from the 200M-step PPO teacher", fontsize=11, color=INK)
a.legend(fontsize=9, loc="lower right", frameon=False)
a.set_ylim(0, 6400)

b = ax[1]
b.loglog([r["heldout_action_mse"] for r in hp], [100 - r["teacher_retention_pct"] for r in hp],
         "o", color=C_HP, ms=7, label="HyperplaneMHL")
b.loglog([r["heldout_action_mse"] for r in fa], [100 - r["teacher_retention_pct"] for r in fa],
         "s", color=C_FAST, ms=7, label="FastMHL")
for r in hp + fa:
    b.annotate(f"{r['total_params']/1000:.0f}k", (r["heldout_action_mse"],
               max(100 - r["teacher_retention_pct"], 0.05)), fontsize=7.5,
               color=MUTED, xytext=(3, 3), textcoords="offset points")
b.set_xlabel("held-out action MSE (behaviour-cloning loss)")
b.set_ylabel("teacher return lost (%)")
b.set_title("B  Does the cloning loss predict the return?\n"
            "labels = parameter count", fontsize=11, color=INK)
b.legend(fontsize=9, frameon=False)

for axx in ax:
    axx.grid(True, color=GRID, lw=0.8); axx.set_axisbelow(True)
    for s in ("top", "right"):
        axx.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axx.spines[s].set_color(GRID)
    axx.tick_params(colors=MUTED, labelsize=9)
    axx.title.set_color(INK)
    axx.xaxis.label.set_color(MUTED); axx.yaxis.label.set_color(MUTED)

fig.tight_layout()
out = os.path.join(HERE, "representability_curve.png")
fig.savefig(out, dpi=155, facecolor="white")
print("wrote", out)
