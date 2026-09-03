import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as ml
from matplotlib.ticker import NullFormatter

# 16k-step runs across every family, all committed real values used in the paper's tables.
# Label style follows Figure 1 (plot_ffn_grid.py): a compact config tag per marker (cells x
# tables "2^{nap}x{tph}t" for the LUT-routing points; a short family word otherwise), tiny
# font with offset-point annotations and leader lines for crowded points, a red dashed vanilla
# baseline, and a grey Pareto frontier line. Colour is by family (this figure's whole point),
# and the bpb axis is natural (lower = better, not inverted).
# (id, params_M, val_bpb, family, tag)
pts = [
    ("0135", 35.79, 1.20144, "vanilla",  "4x MLP"),
    ("0136", 330.7, 1.20567, "straight", "straightfwd"),
    ("0138", 66.9,  1.21249, "outcomp",  "out-compress"),
    # input reprojection (tag = 2^nap x tph tables; H-prefix when H != 4)
    ("0084", 67.4,  1.19866, "reproj",   "2^7x256t"),
    ("0118", 180.6, 1.17431, "reproj",   "2^9x256t"),
    ("0119", 105.1, 1.18386, "reproj",   "2^9x128t"),
    ("0121", 67.4,  1.19142, "reproj",   "2^8x128t"),     # reprojection anchor -> gold star
    ("0126", 39.0,  1.20694, "reproj",   "2^7x64t"),
    ("0127", 48.5,  1.19471, "reproj",   "2^7x128t"),
    ("0128", 48.5,  1.20228, "reproj",   "2^8x64t"),
    ("0129", 105.1, 1.18074, "reproj",   "2^8x256t"),
    ("0130", 105.1, 1.19363, "reproj",   "2^10x64t"),
    ("0131", 67.4,  1.18883, "reproj",   "H2 2^8x128t"),
    ("0132", 67.4,  1.19263, "reproj",   "H8 2^8x128t"),
    ("0133", 180.6, 1.17910, "reproj",   "2^10x128t"),
    ("0137", 67.4,  1.19432, "reproj",   "H1 2^8x128t"),
    ("0153", 39.0,  1.20772, "reproj",   "H2 2^7x64t"),
]
VANILLA = 1.20144
ANCHOR = "0121"

# family -> (colour, marker, size, legend label)
style = {
    "vanilla":  ("#111111", "X", 110, "vanilla 4x MLP FFN (0135)"),
    "straight": ("#9A6700", "s", 70,  "straightforward LUT"),
    "outcomp":  ("#0F7A45", "D", 60,  "output compression"),
    "reproj":   ("#1D4ED8", "o", 52,  "input reprojection"),
}
GOLD = "#ffd43b"

# Per-id label offset (dx, dy in points, ha); Figure-1 convention. Reprojection column at
# ~67M is pushed left to spread the dense reprojection column labels.
OFF = {
    "0135": (-9, 8, "r"),   "0136": (-9, 6, "r"),   "0138": (9, 4, "l"),
    # 39M pair (0126/0153 nearly coincident): split up/down with leaders
    "0126": (-11, 7, "r"),  "0153": (12, -10, "l"),
    # 48.5M pair
    "0127": (-10, 5, "r"),  "0128": (9, 4, "l"),
    # 67.4M column (five points at one x): fan left/right with leaders
    "0084": (12, 6, "l"),   "0137": (-15, 14, "r"),  "0132": (-15, -12, "r"),
    "0121": (14, 8, "l"),   "0131": (12, -7, "l"),
    # 105M column
    "0119": (-11, -5, "r"), "0129": (11, -5, "l"),  "0130": (11, 5, "l"),
    # 180M column
    "0118": (12, -5, "l"),  "0133": (12, 6, "l"),
}
# points that get a thin leader line (crowded columns fanned out)
LEAD = {"0084", "0137", "0132", "0131", "0121", "0153", "0126", "0119", "0129"}

fig, ax = plt.subplots(figsize=(10.0, 6.8))

seen = set()
for rid, p, b, fam, tag in pts:
    if rid == ANCHOR:
        continue
    c, m, s, leg = style[fam]
    ax.scatter(p, b, c=c, marker=m, s=s, edgecolors="white", linewidths=0.6,
               zorder=3, label=(leg if fam not in seen else None))
    seen.add(fam)

# anchor: gold star over its (reprojection) point
arow = next(r for r in pts if r[0] == ANCHOR)
ax.scatter(arow[1], arow[2], c=GOLD, marker="*", s=320, edgecolors="k", linewidths=1.1, zorder=6)

# grey Pareto frontier (lower envelope: best bpb as parameters grow)
best = 9.9; fx = []; fy = []
for rid, p, b, fam, tag in sorted(pts, key=lambda r: r[1]):
    if b < best - 1e-9:
        best = b; fx.append(p); fy.append(b)
ax.plot(fx, fy, "-", color="#adb5bd", lw=1.2, zorder=1)

# compact config-tag annotations (Figure-1 style)
for rid, p, b, fam, tag in pts:
    dx, dy, side = OFF.get(rid, (8, 3, "l"))
    ha = "left" if side == "l" else "right"
    is_anchor = rid == ANCHOR
    txt = f"{tag}\n(anchor)" if is_anchor else tag
    kw = dict(fontsize=(12 if is_anchor else 11),
              fontweight=("bold" if is_anchor else "normal"),
              xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
              color=("#7a5c00" if is_anchor else "#333"))
    if rid in LEAD:
        kw["arrowprops"] = dict(arrowstyle="-", lw=.45, color="#999", shrinkA=0, shrinkB=2)
    ax.annotate(txt, (p, b), **kw)

# vanilla baseline (red dashed, Figure-1 style) + label
ax.axhline(VANILLA, color="#c92a2a", ls="--", lw=1.3, zorder=2)
ax.text(360, VANILLA + 0.0004, "vanilla 4x MLP FFN baseline (1.20144)",
        fontsize=12, color="#c92a2a", ha="right", va="bottom")

ax.set_xscale("log")
ax.set_xlim(31, 400)
ax.set_ylim(1.170, 1.2165)
ax.set_xticks([35, 50, 70, 100, 150, 200, 330])
ax.set_xticklabels(["35", "50", "70", "100", "150", "200", "330"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis="both", labelsize=12)
ax.set_xlabel("Total parameters (M, log scale)", fontsize=14)
ax.set_ylabel("Validation bpb  ↓  (lower is better)", fontsize=14)
ax.set_title("16k-step runs: validation bpb vs. total parameters\n"
             "label = compact config (cells x tables); "
             "the LUT frontier reaches 1.174, below the vanilla baseline",
             fontsize=14)
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

handles = [ml.Line2D([], [], marker=style[f][1], ls="", mfc=style[f][0], mec="white",
                     ms=9, label=style[f][3]) for f in ["vanilla", "straight", "outcomp", "reproj"]]
handles.append(ml.Line2D([], [], marker="*", ls="", mfc=GOLD, mec="k", ms=15,
                         label="anchor exp_n_0121"))
ax.legend(handles=handles, fontsize=12, title_fontsize=12, loc="lower left", framealpha=0.95, title="family")

fig.tight_layout()
fig.savefig("/home/astarostin/projects/ffn-lut-paper/fig16k.pdf", bbox_inches="tight", pad_inches=0.12)
print("saved fig16k.pdf")
