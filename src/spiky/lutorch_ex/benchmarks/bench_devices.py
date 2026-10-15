"""Compare two devices' performance snapshots (bench_snapshot JSON, the same matrix measured on each) and draw the
comparison figures.

    python -m spiky.lutorch_ex.benchmarks.bench_devices rtx5090.json h100.json --out-dir figures_rtx5090_vs_h100

writes, per geometry, two figures of interleaved bars (first device orange, second blue), one row per cartridge, log
axis, each pair labelled with the ratio second / first: wall time (``time_wall_<geometry>.png``: eval forward | train
step) and GPU kernel time (``time_gpu_<geometry>.png``, the dispatch-free time; a value bench_snapshot flagged
``gpu_suspect`` is drawn hollow); then the training step's peak memory for every geometry (``memory_train.png``), and
``comparison.md``, the joined table.

Unlike bench_compare (two snapshots of one machine, judged against its run-to-run noise floor) the two sides are
different hardware: there is no "within noise" here, only the ratio. Wall time is the median of the per-round medians;
its spread between rounds is in each JSON. matplotlib is needed for the figures, not for ``comparison.md``.
"""
from __future__ import annotations

import argparse
import json
import os

FIRST_C, SECOND_C = "#E69F00", "#0072B2"                   # orange / blue (Okabe-Ito: CVD-distinct)
INK, INK2, MUTED, GRID, SURF = "#1f2328", "#57606a", "#8c959f", "#e6e8eb", "#ffffff"
PHASES = [("eval_forward", "eval forward"), ("train_total", "train step (forward + backward)")]


def _load(path):
    rec = json.load(open(path))
    rows = {(r["geometry"], r["cartridge"]): r for r in rec["rows"] if r.get("dtype") == "float32"}
    return rec, rows


def _short(e):
    return e["gpu"].replace("NVIDIA ", "").replace("GeForce ", "").replace(" 80GB HBM3", "")


def _val(r, phase, key):
    t = (r or {}).get(phase)
    if key == "gpu_suspect":     # bench_snapshot's rule, applied to the aggregated cell: kernel time > 1.05 x wall
        return bool(t and "gpu_ms" in t and t["gpu_ms"] > 1.05 * t["median_ms"])
    return None if not t or key not in t else t[key]


def _plt():
    try:
        import matplotlib
    except ImportError as e:
        raise SystemExit("bench_devices: the figures need matplotlib (pip install matplotlib)") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
    return plt


def _label(name):
    return name.replace("LUT", "").replace(" uncertainty_func=softsign", " · softsign")


def _pair_axes(plt, ax, names, a_vals, b_vals, a_hollow, b_hollow, title, unit, n1, n2):
    import math
    h = 0.36
    ys = list(range(len(names)))[::-1]
    vals = [v for v in a_vals + b_vals if v]
    lo, hi = min(vals) / 1.6, max(vals) * 3.0
    for y, a, b, ah, bh in zip(ys, a_vals, b_vals, a_hollow, b_hollow):
        for v, off, col, hollow in ((a, h / 2, FIRST_C, ah), (b, -h / 2, SECOND_C, bh)):
            if v is None:
                ax.text(lo * 1.1, y + off, "—", va="center", fontsize=7, color=MUTED)
                continue
            ax.barh(y + off, v - lo, left=lo, height=h * 0.92, color=SURF if hollow else col, edgecolor=col,
                    lw=1.2 if hollow else 0, zorder=2)
        if a and b:
            ax.text(max(a, b) * 1.12, y, f"×{b / a:.2f}", va="center", fontsize=7, color=INK)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.3)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(unit + " (LOG SCALE)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)


def _legend(fig, plt, n1, n2, hollow_note):
    from matplotlib.patches import Patch
    handles = [Patch(color=FIRST_C, label=n1), Patch(color=SECOND_C, label=n2),
               Patch(color="none", label=f"×r = {n2} / {n1}  (< 1: {n2} faster or smaller)")]
    if hollow_note:
        handles.append(Patch(facecolor=SURF, edgecolor=MUTED, label="hollow = GPU time flagged gpu_suspect "
                                                                     "(profiler-inflated; use the wall time)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.45 / fig.get_figheight()), ncol=4,
               frameon=False, fontsize=8)


def time_figure(plt, ra, rb, geom, key, n1, n2, env_line, path):
    names = [c for (g, c) in ra if g == geom]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.36 * len(names) + 1.9), sharey=True)
    any_hollow = False
    for ax, (ph, phl) in zip(axes, PHASES):
        a = [_val(ra.get((geom, c)), ph, key) for c in names]
        b = [_val(rb.get((geom, c)), ph, key) for c in names]
        ah = [key == "gpu_ms" and bool(_val(ra.get((geom, c)), ph, "gpu_suspect")) for c in names]
        bh = [key == "gpu_ms" and bool(_val(rb.get((geom, c)), ph, "gpu_suspect")) for c in names]
        any_hollow |= any(ah) or any(bh)
        _pair_axes(plt, ax, [_label(c) for c in names], a, b, ah, bh, phl,
                   "median ms" if key == "median_ms" else "GPU kernel ms per step", n1, n2)
    batch = next(r["batch"] for (g, _), r in ra.items() if g == geom)
    what = "wall time (median of the round medians)" if key == "median_ms" else "GPU kernel time per step"
    fig.suptitle(f"{n1} vs {n2} · {what} · {geom}, {batch:,} vectors, float32, defaults", x=0.01, ha="left",
                 y=1 - 0.1 / fig.get_figheight(), va="top", fontweight="bold", fontsize=11)
    _legend(fig, plt, n1, n2, any_hollow)
    fig.text(0.01, 0.005, env_line, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def memory_figure(plt, ra, rb, n1, n2, env_line, path):
    keys = [k for k in ra if _val(ra[k], "train_total", "peak_gib") is not None]
    names = [f"{_label(c)} · {g}" for g, c in keys]
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(keys) + 1.9))
    a = [_val(ra.get(k), "train_total", "peak_gib") * 1024 for k in keys]
    b = [(_val(rb.get(k), "train_total", "peak_gib") or 0) * 1024 or None for k in keys]
    _pair_axes(plt, ax, names, a, b, [False] * len(a), [False] * len(b), "train step peak memory", "peak MiB above the "
               "pre-step allocation", n1, n2)
    fig.suptitle(f"{n1} vs {n2} · training-step peak memory · float32, defaults", x=0.01, ha="left",
                 y=1 - 0.1 / fig.get_figheight(), va="top", fontweight="bold", fontsize=11)
    _legend(fig, plt, n1, n2, False)
    fig.text(0.01, 0.005, env_line, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown(ra, rb, n1, n2, ea, eb) -> str:
    def f(v, s=1.0, fmt=".3f"):
        return "-" if v is None else format(v * s, fmt)
    out = [f"{n1}: {ea['gpu']} ({ea['sm']}), torch {ea['torch']}, driver {ea['driver']}, commit {ea['commit'][:10]}",
           f"{n2}: {eb['gpu']} ({eb['sm']}), torch {eb['torch']}, driver {eb['driver']}, commit {eb['commit'][:10]}",
           "",
           "Wall = median of the per-round medians (ms); GPU = kernel time per step (ms; * = gpu_suspect); "
           "peak = train-step peak MiB above the pre-step allocation.", "",
           f"| geometry | cartridge | eval wall {n1} | eval wall {n2} | ×eval | train wall {n1} | train wall {n2} | "
           f"×train | train GPU {n1} | train GPU {n2} | train peak MiB {n1} | train peak MiB {n2} |",
           "|---" * 12 + "|"]
    for k, r in ra.items():
        s = rb.get(k)
        ea_, eb_ = _val(r, "eval_forward", "median_ms"), _val(s, "eval_forward", "median_ms")
        ta, tb = _val(r, "train_total", "median_ms"), _val(s, "train_total", "median_ms")

        def gpu(x):
            v = _val(x, "train_total", "gpu_ms")
            return "-" if v is None else f"{v:.3f}" + ("*" if _val(x, "train_total", "gpu_suspect") else "")
        out.append(f"| {k[0]} | {k[1]} | {f(ea_)} | {f(eb_)} | {f(eb_ / ea_ if ea_ and eb_ else None, fmt='.2f')} | "
                   f"{f(ta)} | {f(tb)} | {f(tb / ta if ta and tb else None, fmt='.2f')} | {gpu(r)} | {gpu(s)} | "
                   f"{f(_val(r, 'train_total', 'peak_gib'), 1024, '.1f')} | "
                   f"{f(_val(s, 'train_total', 'peak_gib'), 1024, '.1f')} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--names", default="", help="two comma-separated device labels (default: from the JSONs)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    (rec_a, ra), (rec_b, rb) = _load(a.first), _load(a.second)
    ea, eb = rec_a["environment"], rec_b["environment"]
    n1, n2 = a.names.split(",") if a.names else (_short(ea), _short(eb))
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, "comparison.md"), "w").write(markdown(ra, rb, n1, n2, ea, eb))
    if a.no_figures:
        return
    plt = _plt()
    s = rec_a.get("settings", {})
    env_line = (f"{n1}: {ea['gpu']}, torch {ea['torch']}, driver {ea['driver']}, code {ea['commit'][:8]}   ·   "
                f"{n2}: {eb['gpu']}, torch {eb['torch']}, driver {eb['driver']}, code {eb['commit'][:8]}   ·   "
                f"bench_snapshot: fresh process per cell, {s.get('rounds', '?')} rotated rounds × "
                f"{s.get('warmup', '?')} warm-up / {s.get('repeats', '?')} timed, L2 flushed")
    for geom in dict.fromkeys(g for g, _ in ra):
        time_figure(plt, ra, rb, geom, "median_ms", n1, n2, env_line, os.path.join(a.out_dir, f"time_wall_{geom}.png"))
        time_figure(plt, ra, rb, geom, "gpu_ms", n1, n2, env_line, os.path.join(a.out_dir, f"time_gpu_{geom}.png"))
    memory_figure(plt, ra, rb, n1, n2, env_line, os.path.join(a.out_dir, "memory_train.png"))
    print("figures in", a.out_dir)


if __name__ == "__main__":
    main()
