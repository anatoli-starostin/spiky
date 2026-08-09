"""Plots for the ES sweeps. Primary y is always CORR = raw tau - own_null."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PLOTS = os.path.join(RES, "plots")
BG, INK, MUTED, GRID = "#faf9f7", "#2a2622", "#7d7368", "#ddd8d0"
SERIES = ["#4a6fa5", "#c1666b", "#5c8d5a", "#b8860b", "#7b6d8d", "#4a8b8b"]


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(BG)
    ax.grid(alpha=0.18, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11)


def legend(ax, **kw):
    lg = ax.legend(fontsize=8, framealpha=1.0, facecolor=BG, edgecolor=GRID, **kw)
    for t in lg.get_texts():
        t.set_color(INK)
    return lg


def series(runs, key):
    """-> [n_seeds, n_gens] for a per-generation key."""
    return np.array([[h[key] for h in r["hist"]] for r in runs])


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    os.makedirs(PLOTS, exist_ok=True)
    runs = load("es_sweep_arma_bar.json") or load("es_sweep_arma.json")
    gens = np.arange(series(runs, "val_corr").shape[1])
    corr = series(runs, "val_corr")
    out = []

    # ---- 1. headline: CORR per generation, per seed + mean band --------------------
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    for i, r in enumerate(runs):
        ax.plot(gens, corr[i], lw=1.0, alpha=0.55, color=SERIES[i % len(SERIES)],
                label=f"seed {r['seed']}")
    m, sd = corr.mean(0), corr.std(0, ddof=1)
    ax.plot(gens, m, lw=2.4, color=INK, label="mean")
    ax.fill_between(gens, m - sd, m + sd, color=INK, alpha=0.13, lw=0, label="±1 sd")
    ax.axhline(0, ls="--", lw=1.1, color=MUTED)
    ax.annotate("chance (own-null)", xy=(gens[-1] * 0.62, 0.012), fontsize=8, color=MUTED)
    style(ax, "generation", "corrected val  (tau − own_null)",
          f"Arm A ES: corrected held-out score, {len(runs)} seeds")
    legend(ax, loc="lower right")
    fig.tight_layout()
    p1 = os.path.join(PLOTS, "01_corr_per_seed.png")
    fig.savefig(p1, dpi=150, facecolor=BG)
    out.append((p1, f"all {len(runs)} seeds rise from ~0 to +{m[-1]:.3f}; "
                    f"spread stays within ±{sd.max():.3f}"))

    # ---- 2. raw vs own-null vs corrected -------------------------------------------
    raw, null = series(runs, "val"), series(runs, "val_null")
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    for arr, c, lab in ((raw, SERIES[0], "raw val tau"),
                        (null, SERIES[1], "own-null (chance for THIS model)"),
                        (corr, SERIES[2], "corrected = raw − own-null")):
        mm, ss = arr.mean(0), arr.std(0, ddof=1)
        ax.plot(gens, mm, lw=2.0, color=c, label=lab)
        ax.fill_between(gens, mm - ss, mm + ss, color=c, alpha=0.13, lw=0)
    style(ax, "generation", "tau-b",
          "Most of the raw score is the model's own constant-ordering bias")
    legend(ax, loc="center right")
    fig.tight_layout()
    p2 = os.path.join(PLOTS, "02_raw_null_corrected.png")
    fig.savefig(p2, dpi=150, facecolor=BG)
    out.append((p2, f"raw ends +{raw.mean(0)[-1]:.3f} but own-null rises to "
                    f"+{null.mean(0)[-1]:.3f}, leaving only +{corr.mean(0)[-1]:.3f} real"))

    # ---- 3. best-so-far, the plateau -------------------------------------------------
    best = np.maximum.accumulate(corr, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    mm, ss = best.mean(0), best.std(0, ddof=1)
    ax.plot(gens, mm, lw=2.4, color=SERIES[0], label="best-so-far (mean)")
    ax.fill_between(gens, mm - ss, mm + ss, color=SERIES[0], alpha=0.15, lw=0, label="±1 sd")
    g95 = int(np.argmax(mm >= 0.95 * mm[-1]))
    ax.axvline(g95, ls="--", lw=1.2, color=SERIES[1])
    ax.annotate(f"95% of final reached\nby generation {g95}", xy=(g95 + 3, mm[-1] * 0.55),
                fontsize=8.5, color=SERIES[1])
    style(ax, "generation", "best corrected val so far",
          "Diminishing returns: most of the gain is early")
    legend(ax, loc="lower right")
    fig.tight_layout()
    p3 = os.path.join(PLOTS, "03_best_so_far.png")
    fig.savefig(p3, dpi=150, facecolor=BG)
    out.append((p3, f"best-so-far plateaus at +{mm[-1]:.3f}; 95% of it by generation {g95}"))

    # ---- 4. arm/protocol comparison ---------------------------------------------------
    variants = [("es_sweep_arma.json", "arm A (random)", SERIES[0], "-"),
                ("es_sweep_armb.json", "arm B (3600 s STDP)", SERIES[1], "-"),
                ("es_sweep_arma_wo.json", "arm A · weights-only", SERIES[0], "--"),
                ("es_sweep_armb_wo.json", "arm B · weights-only", SERIES[1], "--")]
    have = [(n, l, c, s) for n, l, c, s in variants if load(n)]
    if have:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
        for name, lab, c, ls in have:
            rr = load(name)
            cc = series(rr, "val_corr")
            g = np.arange(cc.shape[1])
            mm, ss = cc.mean(0), cc.std(0, ddof=1)
            ax.plot(g, mm, lw=2.0, color=c, ls=ls, label=f"{lab}  (n={len(rr)})")
            ax.fill_between(g, mm - ss, mm + ss, color=c, alpha=0.10, lw=0)
        ax.axhline(0, ls=":", lw=1.0, color=MUTED)
        style(ax, "generation", "corrected val  (tau − own_null)",
              "Freezing wiring+delays costs more than the reservoir choice")
        legend(ax, loc="lower right")
        fig.tight_layout()
        p4 = os.path.join(PLOTS, "04_arm_comparison.png")
        fig.savefig(p4, dpi=150, facecolor=BG)
        out.append((p4, "solid vs dashed (weights-only) gap exceeds the arm A vs arm B gap"))

    for p, note in out:
        print(f"{p}\n    {note}")


if __name__ == "__main__":
    main()
