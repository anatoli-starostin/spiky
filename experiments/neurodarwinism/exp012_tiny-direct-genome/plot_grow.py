"""exp012 growable nets: given the freedom to grow, evolution shrank.

  A  the size trajectory. Seeded at 8 exc + 2 inh, the pool falls to 4 neurons and 56 synapses
     and then holds there for a thousand rounds. Inhibition is gone by round ~244.
  B  MSE against synapse count for all four winners: the growable net matches the best fixed-
     size result on 40 % of its synapses.
  C  the decomposition — where the error sits in each.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
ARMS = [("mutation only", "full_run", "#7FA8C9", 25.231, 140),
        ("+ crossover", "full_run_crossover", "#59A14F", 23.306, 140),
        ("+ lateral inhib", "full_run_crossover_latinhib", "#E1A03C", 24.271, 134),
        ("growable (seeded\nfrom lateral inhib)", "full_run_grow", "#B4453C", 23.445, 56)]
LEAD = {}
for name, path, *_ in ARMS:
    p = os.path.join(D, path, "final_leader.json")
    LEAD[name] = json.load(open(p))
H = json.load(open(os.path.join(D, "full_run_grow", "g0.json")))
CONST = 34.152

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.4, 5.6))
fig.subplots_adjust(left=0.055, right=0.985, top=0.70, bottom=0.135, wspace=0.34)

# ---------------------------------------------------------------- A the shrink
rnd = np.array([h["rnd"] for h in H])
ax.plot(rnd, [h["n_exc_mean"] for h in H], "-", lw=2.4, color="#4E79A7",
        label="excitatory (pool mean)")
ax.plot(rnd, [h["n_inh_mean"] for h in H], "-", lw=2.2, color=C_T,
        label="inhibitory (pool mean)")
ax.plot(rnd, [h["n_active_mean"] for h in H], "-", lw=1.4, color=INK, alpha=0.55,
        label="total active")
ax.axhline(10, color=MUTE, ls=":", lw=1.5)
ax.text(1490, 10.3, "seeded at 10 (8 exc + 2 inh)", fontsize=8.5, color=MUTE, ha="right")
ax.annotate("inhibition effectively gone\nby round 244", (244, 0.5),
            textcoords="offset points", xytext=(40, 42), fontsize=8.5, color=C_T,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_T, lw=1.1))
ax2 = ax.twinx()
ax2.plot(rnd, [h["n_syn_mean"] for h in H], "--", lw=1.8, color="#B07AA1")
ax2.set_ylabel("synapses (pool mean, dashed)", fontsize=10, color="#B07AA1")
ax2.tick_params(axis="y", labelcolor="#B07AA1", labelsize=9)
ax2.set_ylim(0, 150)
ax2.spines["top"].set_visible(False)
ax.set_ylim(-0.5, 11.5)
ax.set_xlabel("round", fontsize=10)
ax.set_ylabel("active hidden neurons", fontsize=10)
ax.set_title("A · Given room to grow to 50, it shrank to 4\n"
             "seeded at 8 exc + 2 inh and free to reach 40 + 10; the pool falls\n"
             "to 4 excitatory, 0 inhibitory by round ~500 and holds for 1,000 more",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="center right")

# ---------------------------------------------------------------- B cost vs quality
# the three fixed-size winners sit within 6 synapses of each other, so their labels need
# explicit, non-colliding offsets rather than one shared rule
OFF = {"mutation only": ((0, 15), "center"), "+ crossover": ((13, -4), "left"),
       "+ lateral inhib": ((-13, -3), "right"),
       "growable (seeded\nfrom lateral inhib)": ((0, -34), "center")}
for name, path, col, mse, syn in ARMS:
    bx.plot([syn], [mse], "o", ms=13, color=col, mec="white", mew=1.6, zorder=5)
    xy, ha = OFF[name]
    bx.annotate(name, (syn, mse), textcoords="offset points", xytext=xy, fontsize=8.5,
                color=INK, ha=ha, fontweight="bold")
bx.axhline(CONST, color=C_T, ls="--", lw=1.6)
bx.text(40, CONST - 0.7, f"constant predictor {CONST:.1f}", fontsize=9, color=C_T,
        fontweight="bold", ha="left")
bx.annotate("", xy=(60, 23.50), xytext=(130, 24.23),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.6, ls=":"))
bx.text(97, 25.35, "−58 % synapses,  −0.83 MSE", fontsize=8.5, color=INK, ha="center",
        fontweight="bold")
bx.set_xlim(30, 185)
bx.set_ylim(21.6, 35)
bx.set_xlabel("synapses in the final network", fontsize=10)
bx.set_ylabel("held-out MSE", fontsize=10)
bx.set_title("B · It matches the best fixed-size net on 40 % of its synapses\n"
             "23.445 on 56 synapses against 23.306 on 140 — statistically a tie\n"
             "on quality, a 2.5× difference in size",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- C decomposition
lbl3 = ["bias²", "scale error", "residual"]
cols = ["#B07AA1", "#F1A340", "#4E79A7"]
xs = np.arange(len(ARMS))
for k, (name, _p, _c, _m, _s) in enumerate(ARMS):
    G = LEAD[name]
    bot = 0.0
    for j, key in enumerate(("bias2", "scale_err", "resid")):
        v = G[key]
        cx.bar(xs[k], v, 0.55, bottom=bot, color=cols[j], zorder=3, edgecolor="white",
               linewidth=2, label=lbl3[j] if k == 0 else None)
        cx.text(xs[k], bot + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")
        bot += v
    cx.plot([xs[k]], [G["affine_ceiling"]], "*", ms=16, color="#59A14F", mec="white",
            mew=1.0, zorder=7)
    cx.text(xs[k], bot + 0.5, f"{G['heldout_mse']:.2f}", ha="center", fontsize=9.5,
            color=INK, fontweight="bold")
cx.axhline(CONST, color=C_T, ls="--", lw=1.6, zorder=6)
cx.text(-0.45, CONST + 0.5, f"constant  {CONST:.1f}", fontsize=9, color=C_T,
        fontweight="bold", ha="left")
cx.set_xticks(xs)
cx.set_xticklabels(["mutation\nonly", "+ cross-\nover", "+ lateral\ninhib", "growable\n(4 neurons)"],
                   fontsize=8.5)
cx.set_ylim(0, 40)
cx.set_ylabel("held-out MSE", fontsize=10)
cx.set_title("C · Same error profile, a quarter of the machine\n"
             "★ = affine ceiling: 17.0 for the 4-neuron net against 16.7\n"
             "for the 140-synapse one — the information barely moved",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="upper right")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)
ax.spines["right"].set_visible(True)

fig.suptitle("exp012 growable nets — size is a gene, and selection spent it downwards · "
             "fitness = MSE + 0.35·neurons + 0.10·fan-out excess",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_grow.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
