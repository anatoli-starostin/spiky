"""Compare two performance snapshots (bench_snapshot JSON) and draw the comparison figures.

    python -m spiky.lutorch_ex.benchmarks.bench_compare before.json after.json --out-dir figures_before_vs_after

reads two snapshot JSONs (nothing else: no PR text, no benchmark run) and writes five PNGs: interleaved before / after
bars (before orange, after green) for median time and for peak memory, each split into two figures by phase (eval
forward + train total; train forward + train backward), and a ratio strip with every comparable cell (after / before,
log axis, the run-to-run band shaded). ``--csv`` also writes the joined table.

How cells are joined and marked:
- A cell is (class, vectors, dtype, phase, metric).
- Out of memory is an "OOM" stub in the bar's colour, never a zero-length or missing bar.
- Each cell has its own noise floor (``noise_floor``), the run-to-run spread measured on this machine with identical
  code and clocks unlocked: a timing whose before value is under 0.5 ms (launch-bound) +-15%; any other timing at
  8,192 vectors or fewer +-10%; any other timing +-6%; peak memory +-5%. A change within the cell's floor is
  "within noise": the bars draw the floor as a shaded band around the before value, hatch the after-bar, and label
  the ratio in grey with the floor ("x1.03 . within +-10%"); a change beyond it gets the ratio in black.
- A row only the AFTER snapshot has (new functionality: nothing to compare it with) is drawn too, as the after bar
  alone, labelled "NEW" -- no ratio and no noise-floor judgement; it is listed, not plotted, on the ratio strip. A row
  only the BEFORE snapshot has is reported on stdout. Without after-only rows the figures are unchanged.
- The footnote states the protocol and the two environments. Two snapshots from different sessions are not a
  controlled A/B, and the figures say so; ``--same-session`` (both measured in one session, interleaved cell by cell)
  says that instead.

matplotlib is needed only for the figures (``--csv`` alone works without it).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

PHASES = [("eval_forward", "eval forward"), ("train_forward", "train forward"), ("train_backward", "train backward"),
          ("train_total", "train total")]
METRICS = [("median_ms", "time", "median ms"), ("peak_gib", "peak memory", "peak GiB above the pre-step allocation")]
NOISE = 0.05                    # peak memory; the band drawn on the ratio strip


def noise_floor(metric: str, vectors: int, before: float) -> float:
    """The cell's run-to-run noise floor (module docstring): the relative change that identical code shows between
    runs on this machine, clocks unlocked."""
    if metric != "time":
        return NOISE
    if before < 0.5:
        return 0.15                 # launch-bound cells
    return 0.10 if vectors <= 8192 else 0.06


#: Row order of the figures: the reference Manifesto family, the fused families, Confidence, Quantised, Deployed.
ORDER = ["ManifestoHardLUT", "ManifestoHardLUT uncertainty_func=softsign",
         "ManifestoSoftLUT", "ManifestoSoftLUT uncertainty_func=softsign",
         "FusedManifestoHardLUT", "FusedManifestoHardLUT uncertainty_func=softsign",
         "FusedManifestoSoftLUT", "FusedManifestoSoftLUT uncertainty_func=softsign",
         "ConfidenceLUT n=1", "ConfidenceLUT n=2", "FusedConfidenceLUT n=1", "FusedConfidenceLUT n=2",
         "QuantisedConfidenceLUT n=1", "QuantisedConfidenceLUT n=2",
         "FusedQuantisedConfidenceLUT n=1", "FusedQuantisedConfidenceLUT n=2",
         "DeployedQuantisedConfidenceLUT", "DeployedQuantisedConfidenceLUT n=1"]
BEFORE_C, AFTER_C = "#E69F00", "#009E73"                  # orange / green (Okabe-Ito: CVD-distinct)
IMPROVE, REGRESS, NOISE_C = "#0072B2", "#D55E00", "#8c959f"
INK, INK2, MUTED, GRID, SURF, BAND = "#1f2328", "#57606a", "#8c959f", "#e6e8eb", "#ffffff", "#eef1f4"


def _load(path):
    rec = json.load(open(path))
    rows = {}
    for r in rec["rows"]:
        cells = {}
        for ph, _ in PHASES:
            p = r.get(ph)
            if p and p.get("median_ms") is not None:
                cells[ph] = p
            elif r.get("status") == "out of memory" and ph != "eval_forward":
                cells[ph] = "OOM"
        rows[(r["cartridge"], r["batch"], r["dtype"])] = (cells, r.get("status", ""))
    return rec, rows


def join(before_path, after_path):
    """(cells, before_only, after_only, recs): the comparable cells, and the rows only one snapshot has."""
    rb, before = _load(before_path)
    ra, after = _load(after_path)
    cells = []
    for key in sorted(set(before) & set(after), key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1], k[2])):
        bc, _ = before[key]
        ac, _ = after[key]
        for ph, phl in PHASES:
            b, a = bc.get(ph), ac.get(ph)
            if b is None and a is None:
                continue                                  # e.g. Deployed has no training phases
            for m, ml, _ in METRICS:
                c = dict(cls=key[0], vectors=key[1], dtype=key[2], phase=phl, metric=ml, before=None, after=None,
                         ratio=None, noise=None, status="")
                if b == "OOM" or a == "OOM" or b is None or a is None:
                    c.update(before=None if b in ("OOM", None) else b[m], after=None if a in ("OOM", None) else a[m],
                             status=f"{'OOM' if b == 'OOM' else 'missing' if b is None else 'fits'} before, "
                                    f"{'OOM' if a == 'OOM' else 'missing' if a is None else 'fits'} after")
                else:
                    bv, av = b[m], a[m]
                    r = av / bv if bv > 0 else float("nan")
                    floor = noise_floor(ml, key[1], bv)
                    noise = abs(r - 1) <= floor
                    c.update(before=bv, after=av, ratio=r, noise=floor,
                             status="within noise" if noise else ("regression" if r > 1 else "improvement"))
                cells.append(c)
    return cells, sorted(set(before) - set(after)), sorted(set(after) - set(before)), (rb, ra)


def new_cells(after_path, after_only):
    """Cells of the rows only the after snapshot has (new functionality), one per phase the row measured and metric:
    after value only, no ratio, no noise floor, status "new (after only)"."""
    _, after = _load(after_path)
    cells = []
    for key in sorted(after_only, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1], k[2])):
        ac, _ = after[key]
        for ph, phl in PHASES:
            a = ac.get(ph)
            if a is None:
                continue
            for m, ml, _ in METRICS:
                cells.append(dict(cls=key[0], vectors=key[1], dtype=key[2], phase=phl, metric=ml, before=None,
                                  after=None if a == "OOM" else a[m], ratio=None, noise=None,
                                  status="new (after only)" if a != "OOM" else "new (after only), OOM"))
    return cells


def _order(c):
    return (ORDER.index(c["cls"]) if c["cls"] in ORDER else 99, c["vectors"], c["dtype"])


def _env(rec, path):
    e = rec["environment"]
    return (f"{os.path.basename(path)}: code {e['commit'][:8]}, {e.get('gpu', '?')}, torch {e.get('torch', '?')}, "
            f"driver {e.get('driver', '?')}, clocks {'locked' if e.get('clocks_locked') else 'unlocked'}")


def _protocol(rec):
    s = rec.get("settings", {})
    return (f"{s.get('rounds', '?')} rotated rounds × {s.get('warmup', '?')} warm-up / "
            f"{s.get('repeats', '?')} timed")


def _plt():
    try:
        import matplotlib
    except ImportError as e:
        raise SystemExit("bench_compare: the figures need matplotlib (pip install matplotlib); "
                         "--csv alone runs without it") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "hatch.color": SURF, "hatch.linewidth": 1.2})
    return plt


def _label(c):
    s = c["cls"].replace("LUT", "").replace(" uncertainty_func=softsign", "·softsign")
    v = f"{c['vectors'] // 1024}k" if c["vectors"] % 1024 == 0 else str(c["vectors"])
    return f"{s} · {c['dtype'].replace('float32', 'fp32').replace('bfloat16', 'bf16')} · {v}"


def _bars(plt, cells, metric, phases, title, unit, lo, hi, caveat, names, path):
    from matplotlib.patches import Patch
    sub = [[c for c in cells if c["metric"] == metric and c["phase"] == p] for p in phases]
    n = max(len(s) for s in sub)
    H = 0.27 * n + 2.4
    fig, axes = plt.subplots(1, len(phases), figsize=(19, H), gridspec_kw=dict(wspace=0.9))
    for ax, rows, phase in zip(axes, sub, phases):
        h, gap = 0.36, 0.28
        ys = []
        for i, c in enumerate(rows):
            yc = -i * (2 * h + gap)
            ys.append(yc)
            noise = c["status"] == "within noise"
            if c["status"].startswith("new (after only)"):  # new functionality: the after bar alone, no judgement
                if c["after"] is None:
                    ax.text(lo * 1.15, yc - h / 2, "OOM", va="center", ha="left", fontsize=7, fontweight="bold",
                            color=AFTER_C)
                else:
                    ax.barh(yc - h / 2, c["after"] - lo, left=lo, height=h * 0.94, color=AFTER_C, lw=0, zorder=2)
                ax.text(lo * 1.15, yc + h / 2, "NEW", va="center", ha="left", fontsize=7, fontweight="bold", color=INK)
                if c["after"] is not None:
                    ax.text(c["after"] * 1.12, yc, "NEW · no before, not ratio'd", va="center", fontsize=6.5,
                            color=INK, fontstyle="italic")
                continue
            if c["noise"] is not None and c["before"]:      # the cell's noise floor around the before value
                ax.add_patch(plt.Rectangle((c["before"] * (1 - c["noise"]), yc - h - 0.02),
                                           c["before"] * 2 * c["noise"], 2 * h + 0.04,
                                           color=BAND, ec=MUTED, lw=0.4, zorder=0))
            bars = ((c["before"], BEFORE_C, h / 2, c["status"].startswith("OOM before"), None),
                    (c["after"], AFTER_C, -h / 2, c["status"].endswith("OOM after"), "////" if noise else None))
            for val, col, off, oom, hatch in bars:
                if val is None:
                    ax.text(lo * 1.15, yc + off, "OOM" if oom else "—", va="center", ha="left", fontsize=7,
                            fontweight="bold", color=col)
                    continue
                ax.barh(yc + off, val - lo, left=lo, height=h * 0.94, color=col, hatch=hatch,
                        edgecolor=SURF if hatch else col, lw=0, zorder=2)
            if c["ratio"] is not None and not noise:
                ax.text(max(c["before"], c["after"]) * 1.12, yc, f"×{c['ratio']:.2f}", va="center", fontsize=7,
                        color=INK, fontweight="bold")
            elif c["ratio"] is not None:
                ax.text(max(c["before"], c["after"]) * (1 + c["noise"]) * 1.06, yc,
                        f"×{c['ratio']:.2f} · within ±{c['noise']:.0%}", va="center", fontsize=6.5, color=MUTED)
            elif c["status"] == "OOM before, fits after":
                ax.text(c["after"] * 1.12, yc, "OOM → fits", va="center", fontsize=7, color=INK, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_yticks(ys)
        ax.set_yticklabels([_label(c) for c in rows], fontsize=7.3)
        ax.set_ylim(ys[-1] - 2 * h, ys[0] + 2 * h)
        ax.set_xlabel(unit + " (LOG SCALE)")
        ax.set_title(phase.capitalize(), loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(title + "   ·   LOG axis: bar lengths are not proportional to the values — read the ×ratio labels",
                 x=0.01, ha="left", y=1 - 0.12 / H, fontsize=12, fontweight="bold")
    handles = [Patch(color=BEFORE_C, label=names[0]), Patch(color=AFTER_C, label=names[1]),
               Patch(facecolor=AFTER_C, hatch="////", edgecolor=SURF,
                     label="after, within the cell's noise floor"),
               Patch(facecolor=BAND, edgecolor=MUTED, lw=0.4,
                     label="the cell's noise floor around the before value (±15% < 0.5 ms; ±10% ≤ 8k; ±6%; "
                           "memory ±5%)"),
               Patch(color="none", label="×r = after / before: black beyond noise, grey within")]
    if any(c["status"].startswith("new (after only)") for s in sub for c in s):
        handles.append(Patch(color="none", label="NEW = a row only the after snapshot has (new functionality): the "
                                                 "after bar alone, no ratio"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.42 / H), ncol=3, frameon=False,
               fontsize=8)
    fig.text(0.01, 0.06 / H, caveat, fontsize=7, color=INK2, va="bottom")
    fig.subplots_adjust(top=1 - 1.2 / H, bottom=1.45 / H, left=0.205, right=0.975)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _strip(plt, cells, title, caveat, path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    groups = [(p, m) for _, m, _ in METRICS for _, p in PHASES]
    fig, ax = plt.subplots(figsize=(12.5, 6.9))
    ax.axvspan(1 - NOISE, 1 + NOISE, color=BAND, zorder=0)
    ax.axvline(1, color=MUTED, lw=1, zorder=1)
    for gi, (ph, m) in enumerate(groups):
        y0 = len(groups) - 1 - gi
        rs = sorted((c for c in cells if c["phase"] == ph and c["metric"] == m and c["ratio"] is not None),
                    key=lambda c: c["ratio"])
        for j, c in enumerate(rs):
            col = NOISE_C if c["status"] == "within noise" else (REGRESS if c["status"] == "regression" else IMPROVE)
            ax.plot([c["ratio"]], [y0 + ((j % 5) - 2) * 0.11], "o", ms=6.5, color=col, mec=SURF, mew=1, zorder=3)
        regs = [c for c in rs if c["status"] == "regression"]
        if regs:
            ax.text(max(c["ratio"] for c in regs) * 1.12, y0, "\n".join(f"{_label(c)} ×{c['ratio']:.2f}" for c in regs),
                    va="center", fontsize=6.6, color=INK2, linespacing=1.1)
        if any(c["phase"] == ph and c["metric"] == m and c["status"] == "OOM before, fits after" for c in cells):
            ax.text(0.012, y0, "◀ OOM → fits", transform=ax.get_yaxis_transform(), fontsize=7, color=IMPROVE,
                    va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.005, 60)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{ph} · {m}" for ph, m in reversed(groups)])
    ax.set_xlabel("after / before   (< 1: faster or smaller;  > 1: slower or larger;  LOG SCALE)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=IMPROVE, ms=6.5, label="improvement (beyond noise)"),
                       Line2D([], [], marker="o", ls="", color=REGRESS, ms=6.5, label="regression (beyond noise)"),
                       Line2D([], [], marker="o", ls="", color=NOISE_C, ms=6.5, label="within noise"),
                       Patch(color=BAND, label=f"±{NOISE:.0%} band (each cell is judged by its own floor)")],
              loc="upper left", bbox_to_anchor=(0, 1.1), ncol=4, frameon=False, fontsize=8)
    ax.set_title(title, loc="left", pad=28, fontsize=12, fontweight="bold")
    new = sorted({_label(c) for c in cells if c["status"].startswith("new (after only)")})
    if new:
        ax.text(0.995, 0.02, "NEW, after only (no ratio, not plotted):\n" + "\n".join(new), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.6, color=INK2, linespacing=1.15)
    fig.text(0.01, 0.005, caveat, fontsize=7, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.87, bottom=0.21)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--csv", action="store_true", help="also write comparison.csv (the joined cells)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--same-session", action="store_true",
                    help="the two snapshots were measured in one session, interleaved cell by cell "
                         "(the footnote says so)")
    a = ap.parse_args(argv)
    cells, before_only, after_only, (rb, ra) = join(a.before, a.after)
    extra = new_cells(a.after, after_only)
    if extra:                                        # stable merge in the figures' row order; no-op without new rows
        cells = sorted(cells + extra, key=_order)
    os.makedirs(a.out_dir, exist_ok=True)
    nb, na = (os.path.splitext(os.path.basename(p))[0] for p in (a.before, a.after))
    if a.csv:
        with open(os.path.join(a.out_dir, "comparison.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cells[0]))
            w.writeheader()
            w.writerows(cells)
    print(f"{len(cells) - len(extra)} comparable values; before only: {len(before_only)}; after only: {len(after_only)}"
          + (f" ({len(extra)} new-functionality values drawn as after-only bars)" if extra else ""))
    for k in before_only:
        print("  before only:", *k)
    for k in after_only:
        print("  after only: ", *k)
    if a.no_figures:
        return
    plt = _plt()
    pb, pa = _protocol(rb), _protocol(ra)
    proto = pb if pb == pa else f"before {pb}, after {pa}"
    caveat = (f"Before = {_env(rb, a.before)}.\nAfter = {_env(ra, a.after)}.\n"
              "bench_snapshot.py (each class's defaults, one fresh process per cell, Dynamo reset, "
              f"{proto}, L2 flushed, median; peak above the pre-step allocation), "
              + ("one session, interleaved cell by cell (same conditions)." if a.same_session
                 else "different sessions: not a controlled A/B.")
              + "\nHatched / grey = within the cell's noise floor "
              "(±15% timings < 0.5 ms, ±10% timings at ≤ 8k vectors, ±6% other timings, ±5% memory).")
    names = (f"before ({nb})", f"after ({na})")
    t = f"Time per step, {nb} vs {na} — every comparable cell"
    m = f"Peak memory, {nb} vs {na} — every comparable cell"
    out = lambda f: os.path.join(a.out_dir, f)   # noqa: E731
    _bars(plt, cells, "time", ["eval forward", "train total"], t + " (1/2)", "median ms", 0.1, 300, caveat, names,
          out("bars_time_A.png"))
    _bars(plt, cells, "time", ["train forward", "train backward"], t + " (2/2)", "median ms", 0.1, 300, caveat, names,
          out("bars_time_B.png"))
    _bars(plt, cells, "peak memory", ["eval forward", "train total"], m + " (1/2)",
          "peak GiB above the pre-step allocation", 0.005, 40, caveat, names, out("bars_memory_A.png"))
    _bars(plt, cells, "peak memory", ["train forward", "train backward"], m + " (2/2)",
          "peak GiB above the pre-step allocation", 0.005, 40, caveat, names, out("bars_memory_B.png"))
    _strip(plt, cells, f"Every comparable cell: relative change, {nb} → {na}", caveat, out("ratio_strip.png"))
    print("figures in", a.out_dir)


if __name__ == "__main__":
    main()
