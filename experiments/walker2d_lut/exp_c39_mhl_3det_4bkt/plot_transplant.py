"""exp_c39 diagnosis — the transplant ladder, which is the punchline.

Run in the SPIKY venv (matplotlib).

Nothing measurable in the finished checkpoints separates the seed that took off from the
two that did not, and at init they are indistinguishable on every aggregate. So instead of
looking harder at the artefacts, swap the pieces: the trainer is deterministic from
`PRNGKey(seed)` and the actor's init consumes exactly one key, so the INIT and the RL
TRAJECTORY can be exchanged independently, and the init itself can be split into its
front-end half (delay, w_raw -- what the addressing does) and its table half (0.1*randn --
the initial action values).

LEFT — RL stream 0, which originally scored 891. Each bar swaps in more of the winner's
init. The ladder is monotone and it ends four times higher than it started, on the SAME RL
stream: exploration noise, replay order, critic init and env resets are all unchanged.

RIGHT — the same substitution on the other two streams. The winner's init rescues both
losing streams; a loser's init destroys the winning stream.

Usage:
  python plot_transplant.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, RED, AMBER = "#1f9e5a", "#c0392b", "#e08a1e"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1

# RL stream 0 — the ladder. (label, front-end src, table src, score, full)
LADDER = [
    ("original\ns0", "s0", "s0", 890.8, 0),
    ("E\ntable only", "s0", "s2", 1244.4, 0),
    ("D\nfront-end only", "s2", "s0", 1867.5, 3),
    ("A\nboth halves", "s2", "s2", 4001.7, 87),
]
# The other streams. (label, description, score, full, colour)
OTHERS = [
    ("s1 original", "own init", 982.3, 0, RED),
    ("F", "winner FRONT-END\nonly, stream 1", 3860.3, 97, GREEN),
    ("C", "winner init\nboth halves, stream 1", 3455.0, 69, GREEN),
    ("s2 original", "own init", 4217.3, 100, GREEN),
    ("B", "LOSER init on the\nWINNER's stream", 970.8, 0, RED),
]


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    for ax in axes:
        style(ax)
        ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color="#eb6834", alpha=0.12,
                   zorder=1)
        ax.axhline(BASE_M, color="#eb6834", linewidth=1.8, zorder=2)
        ax.set_ylim(0, 5000)

    # ---- LEFT: the ladder on one RL stream -------------------------------
    ax = axes[0]
    xs = np.arange(len(LADDER))
    vals = [v for _, _, _, v, _ in LADDER]
    cols = [RED, RED, AMBER, GREEN]
    ax.bar(xs, vals, width=0.6, color=cols, alpha=0.9, zorder=4,
           edgecolor="white", linewidth=1.4)
    for x, (lab, fe, tb, v, full) in zip(xs, LADDER):
        ax.annotate(f"{v:.0f}\n{full}/100 full", xy=(x, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", color=INK,
                    fontsize=8.4, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\nfe={fe} tab={tb}" for l, fe, tb, _, _ in LADDER],
                       fontsize=7.8)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("RL stream 0 held FIXED — only the init is swapped",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.annotate("baseline 4308 ± 500", xy=(-0.45, BASE_M - BASE_SD), xytext=(0, -12),
                textcoords="offset points", color="#eb6834", fontsize=8.2,
                ha="left", va="top", fontweight="bold")

    # ---- RIGHT: the other streams ----------------------------------------
    ax = axes[1]
    xs = np.arange(len(OTHERS))
    vals = [v for _, _, v, _, _ in OTHERS]
    cols = [c for _, _, _, _, c in OTHERS]
    ax.bar(xs, vals, width=0.6, color=cols, alpha=0.9, zorder=4,
           edgecolor="white", linewidth=1.4)
    for x, (lab, desc, v, full, c) in zip(xs, OTHERS):
        ax.annotate(f"{v:.0f}\n{full}/100", xy=(x, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", color=INK,
                    fontsize=8.4, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n{d}" for l, d, _, _, _ in OTHERS], fontsize=7.6)
    ax.axvline(2.5, color=GRID, linewidth=1.4)
    ax.set_title("Streams 1 and 2 — the winner's init rescues, a loser's init destroys",
                 color=INK, fontsize=11, loc="left", pad=10)

    fig.suptitle("exp_c39 — the outcome follows the ACTOR INIT, not the RL trajectory",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             "Every run here is a real, self-consistent training run: only the actor's "
             "starting parameters are exchanged. Critic init, env resets, exploration "
             "noise and replay sampling all stay with the stream's own seed.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "The front-end half (delay, w_raw — what the addressing does) carries most of "
             "it: alone it lifts 891→1868 and 982→3860. The table half alone lifts "
             "891→1244. Together they reach 4002, so they also interact.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c39_diag_transplant.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
