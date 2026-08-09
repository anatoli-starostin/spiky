"""Aggregate sweep_*.json into the RESULTS.md tables and the sample-efficiency figure.

Aggregation rule, stated once so the tables are readable: for each matrix cell we take the
BEST learning rate by validation error, then report both seeds. "best" always means the
best-validation checkpoint (early stopping), because the surrogate gradient does not vanish
at the exact solution and the final-step model is often far worse than the best one — both
numbers are shown so that drift is visible rather than hidden.
"""
import argparse
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

BG = "#faf9f7"
INK = "#2a2622"
MUTED = "#7d7368"
SERIES = ["#4a6fa5", "#c1666b", "#5c8d5a", "#b8860b", "#7b6d8d", "#4a8b8b"]


def key(r):
    c = r["cfg"]
    return (c["neuron"], c["variant"], c["scope"], c["target"],
            c.get("decode", "affine"), c["perturb"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(RES, "sweep_main.json"))
    ap.add_argument("--fig", default=os.path.join(RES, "sample_efficiency.png"))
    a = ap.parse_args()
    R = json.load(open(a.json))
    astd = R[0]["action_std"]
    by = collections.defaultdict(list)
    for r in R:
        by[key(r)].append(r)

    rows = []
    for k, runs in by.items():
        # best lr by mean-over-seeds best-val
        bylr = collections.defaultdict(list)
        for r in runs:
            bylr[r["cfg"]["lr"]].append(r)
        best_lr = min(bylr, key=lambda lr: np.mean([x["best_val"]["mean"] for x in bylr[lr]]))
        sel = bylr[best_lr]
        rows.append(dict(
            neuron=k[0], variant=k[1], scope=k[2], target=k[3], decode=k[4], perturb=k[5],
            lr=best_lr, n_par=sel[0]["n_trainable"],
            base=np.mean([x["base_val"]["mean"] for x in sel]),
            base_norm=np.mean([x["base_val_norm"] for x in sel]),
            best=np.mean([x["best_val"]["mean"] for x in sel]),
            best_norm=np.mean([x["best_val_norm"] for x in sel]),
            best_max=np.mean([x["best_val"]["max"] for x in sel]),
            best_train=np.mean([x["best_train"]["mean"] for x in sel]),
            final_norm=np.mean([x["final_val_norm"] for x in sel]),
            pairs=int(np.mean([x["best_pairs"] for x in sel])),
            seeds=[round(x["best_val_norm"], 5) for x in sel],
            nospike=np.mean([x["best_val"]["nospike"] for x in sel]),
            clamped=int(np.mean([x["delay_clamped"] for x in sel])),
        ))

    def table(sub, title, show_decode=True):
        print(f"\n### {title}\n")
        hdr = "| variant | scope | target |" + (" decode |" if show_decode else "") + \
              " params | lr | base (norm) | **best (norm)** | best max | train | " \
              "final (norm) | pairs to best | seeds |"
        print(hdr)
        print("|" + "---|" * (hdr.count("|") - 1))
        for r in sorted(sub, key=lambda z: (z["variant"], z["scope"], z["target"],
                                            z["decode"])):
            dec = f" {r['decode']} |" if show_decode else ""
            print(f"| {r['variant']} | {r['scope']} | {r['target']} |{dec} "
                  f"{r['n_par']:,} | {r['lr']:g} | {r['base_norm']:.4f} | "
                  f"**{r['best_norm']:.4f}** | {r['best_max']:.3f} | "
                  f"{r['best_train'] / r['best'] if r['best'] else 0:.2f}x | "
                  f"{r['final_norm']:.4f} | {r['pairs']:,} | "
                  f"{r['seeds'][0]:.4f} / {r['seeds'][-1]:.4f} |")

    print(f"action std = {astd:.4f};  'norm' = mean |a_spiking - a_LUT| / action std")
    table([r for r in rows if r["neuron"] == "exact" and not r["perturb"]],
          "Exact neuron — VERIFICATION cells", show_decode=False)
    table([r for r in rows if r["neuron"] == "lif"], "Hardware LIF — the learning cells")
    table([r for r in rows if r["perturb"]], "Recovery probe (exact neuron, perturbed init)",
          show_decode=False)
    print("\n(recovery-probe rows carry their sigma in the `scope` slot below)")
    for r in sorted([r for r in rows if r["perturb"]], key=lambda z: (z["perturb"], z["variant"])):
        print(f"  sigma={r['perturb']}  variant {r['variant']}  base {r['base_norm']:.4f} "
              f"-> best {r['best_norm']:.4f}  (lr {r['lr']:g}, {r['pairs']:,} pairs)")

    # ---- sample-efficiency figure -------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                     # pragma: no cover
        print(f"\n(no figure: {e})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=BG)
    for ax, var in zip(axes, ("D", "W")):
        ax.set_facecolor(BG)
        ci = 0
        for scope in ("races", "tau", "weights"):
            for decode in ("affine", "corrected"):
                runs = [r for r in R if r["cfg"]["neuron"] == "lif"
                        and r["cfg"]["variant"] == var and r["cfg"]["scope"] == scope
                        and r["cfg"]["target"] == "action"
                        and r["cfg"].get("decode") == decode]
                if not runs:
                    continue
                bylr = collections.defaultdict(list)
                for r in runs:
                    bylr[r["cfg"]["lr"]].append(r)
                blr = min(bylr, key=lambda lr: np.mean([x["best_val"]["mean"]
                                                        for x in bylr[lr]]))
                sel = bylr[blr]
                pts = collections.defaultdict(list)
                for r in sel:
                    pts[0].append(r["base_val_norm"])
                    for c in r["curve"]:
                        pts[c["pairs"]].append(c["best_so_far"] / r["action_std"])
                xs = sorted(pts)
                ys = [np.mean(pts[x]) for x in xs]
                ax.plot([max(x, 500) for x in xs], ys, lw=2, color=SERIES[ci % len(SERIES)],
                        label=f"{scope} / {decode}")
                ci += 1
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distillation pairs seen", color=MUTED, fontsize=9)
        ax.set_title(f"variant {var} ({'delay' if var == 'D' else 'weight'}-coded)",
                     color=INK, fontsize=11)
        ax.grid(alpha=0.18, lw=0.6)
        for s in ax.spines.values():
            s.set_color("#ddd8d0")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if v >= 0.01 else f"{v:.0e}"))
        if var == "D":
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:g}" if 0.2 <= v < 1 else ""))
            ax.tick_params(axis="y", which="minor", labelsize=7, colors=MUTED)
        else:
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].axhline(0.4344, ls="--", lw=1.2, color=MUTED)
    axes[0].annotate("floor for any decode of $t_f$ alone (teacher front-end): 0.434",
                     xy=(5.5e2, 0.452), fontsize=7.5, color=MUTED)
    axes[1].annotate("affine decode — flat, training never helps", xy=(1.5e3, 0.075),
                     fontsize=8.5, color=INK)
    axes[1].annotate("analytic 'corrected' decode:\nteacher-exact with ZERO training",
                     xy=(1.5e3, 3e-5), fontsize=8.5, color=INK)
    leg = axes[0].legend(fontsize=8, framealpha=1.0, facecolor=BG, edgecolor="#ddd8d0",
                         loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle("Hardware-LIF student: accuracy vs distillation pairs seen "
                 "(best-val checkpoint, action target)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=150, facecolor=BG)
    print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
