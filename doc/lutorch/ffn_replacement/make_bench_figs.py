import json, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# Matched-layout FFN-slot phase-breakdown figures for the RTX 5090 and H100, built from the two
# committed results.json files (copied verbatim into bench_data/ so this regenerates without a
# live checkout). The two committed JSONs use DIFFERENT schemas; both are normalised here to the
# SAME horizontal stacked-bar layout. Every stage value is read from the committed data --
# nothing is invented (the only derived quantity is "other" = slot total minus the summed named
# stages, which is ~0 for the H100 routed bars).
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "bench_data")

# fixed stage order + shared colour scheme (matches gpustar's original 5090 figure).
# The residual "other" segment is intentionally NOT drawn (removed from bars + legend); the
# per-bar TOTAL annotation still shows the true committed slot total, not the summed named stages.
ORDER = ["Linear 384->1536 + GELU", "Linear 1536->384",
         "compress 384->192", "routing + gather (fused)", "decompress 192->384"]
COL = {"Linear 384->1536 + GELU": "#4C72B0", "Linear 1536->384": "#A6C0E0",
       "compress 384->192": "#DD8452", "routing + gather (fused)": "#C44E52",
       "decompress 192->384": "#EFC08A", "other": "#BBBBBB"}

def bars_5090():
    d = json.load(open(f"{D}/results_5090.json"))["models"]
    out = []
    for k in ["vanilla", "exp_n_0126", "exp_n_0127", "exp_n_0128"]:
        m = d[k]
        out.append((k, {lab: v for lab, v in m["phases"]}, m["total"], m.get("vs_vanilla")))
    return out

def bars_h100():
    d = json.load(open(f"{D}/results_h100.json"))
    va = d["vanilla"]; vtot = va["slot_eager"]
    out = [("vanilla", {"Linear 384->1536 + GELU": va["up"] + va["gelu"],
                        "Linear 1536->384": va["down"],
                        "other": vtot - (va["up"] + va["gelu"] + va["down"])}, vtot, None)]
    for k in ["0126", "0127", "0128"]:
        r = d["routed"][k]; slot = r["slot"]
        named = r["compress"] + r["fused_route_gather"] + r["decompress"]
        out.append((f"exp_n_{k}",
                    {"compress 384->192": r["compress"],
                     "routing + gather (fused)": r["fused_route_gather"],
                     "decompress 192->384": r["decompress"],
                     "other": slot - named}, slot, r["vs_vanilla"]))
    return out

# Draw one card's horizontal stacked-bar breakdown onto a given axis. Each panel keeps its own
# x-range (xmax) so the within-card stage detail and the LUT/vanilla ratio stay legible; the two
# panels share the stage colour scheme and a single figure-level legend.
def draw(ax, bars, title, xmax):
    ys = list(range(len(bars)))[::-1]   # first bar (vanilla) at the top
    ylabels = []
    for y, (name, stages, total, ratio) in zip(ys, bars):
        left = 0.0
        for st in ORDER:
            if st in stages and stages[st] > 1e-6:
                w = stages[st]
                ax.barh(y, w, left=left, height=0.60, color=COL[st], edgecolor="white", linewidth=0.7)
                if w > 0.018 * xmax:
                    ax.text(left + w / 2, y, f"{w:.3f}", ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")
                left += w
        ax.text(left + 0.008 * xmax, y, f"{total:.3f}", ha="left", va="center",
                fontsize=9, fontweight="bold", color="#333")
        rtxt = "" if ratio is None else f"\n({ratio:.2f}$\\times$)"
        disp = {"vanilla": "vanilla dense"}.get(name, name)
        ylabels.append(f"{disp}{rtxt}")
    ax.set_yticks(range(len(bars))); ax.set_yticklabels(ylabels[::-1], fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("FFN-slot time (ms/call)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.5)

b5 = bars_5090(); bh = bars_h100()
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6))
draw(axL, b5, "RTX 5090", 0.375)
draw(axR, bh, "H100", 0.310)
handles = [mp.Patch(color=COL[s], label=s) for s in ORDER]
fig.legend(handles=handles, fontsize=8.5, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.01), framealpha=0.95)
fig.suptitle("FFN-slot phase breakdown (batch 48 x 512 = 24,576 tokens, bf16)", fontsize=13)
fig.tight_layout(rect=[0, 0.10, 1, 0.95])
fig.savefig(os.path.join(BASE, "bench_combined.pdf"),
            bbox_inches="tight", pad_inches=0.08)
print("saved bench_combined.pdf")
print("5090 stages:", {n: {k: round(v, 5) for k, v in s.items()} for n, s, t, r in b5})
print("H100 stages:", {n: {k: round(v, 5) for k, v in s.items()} for n, s, t, r in bh})
