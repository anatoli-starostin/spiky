#!/usr/bin/env python3
"""
Read lutorch_profile_results.md (summary table with forward_ms, backward_ms, optimizer_step_ms)
and plot forward vs backward timings by mode (backend, smooth, n_alt).

Produces one figure per (smooth, n_alt) with backends on x-axis and grouped bars for forward/backward.
Saves to plot_lutorch_profile_<smooth>_nalt<1|3|all>.png in the same dir as the results file.

Requires: matplotlib. Run from repo root with PYTHONPATH=src.
"""
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    raise SystemExit(1)


def parse_results_table(path: Path) -> list[dict]:
    """Parse the markdown summary table into list of row dicts."""
    text = path.read_text()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    start = None
    for i, line in enumerate(lines):
        if line.startswith("| backend |") and "forward_ms" in line:
            start = i
            break
    if start is None:
        return []
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 7:
            continue
        rows.append({
            "backend": parts[0],
            "smooth": parts[1] == "True",
            "n_alt": parts[2],
            "forward_ms": float(parts[3]),
            "backward_ms": float(parts[4]),
            "optimizer_step_ms": float(parts[5]),
            "elapsed_s": float(parts[6]),
        })
    return rows


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent.parent
    for path in [root / "lutorch_profile_results.md", Path("lutorch_profile_results.md"), root / "src/spiky/lutorch/tests/lutorch_profile_results.md"]:
        if path.exists():
            break
    else:
        print("lutorch_profile_results.md not found")
        return 1

    rows = parse_results_table(path)
    if not rows:
        print("No table rows found in", path)
        return 1

    backends = list(dict.fromkeys(r["backend"] for r in rows))
    n_backends = len(backends)
    x = np.arange(n_backends)
    width = 0.35

    y_max = max(max(r["forward_ms"], r["backward_ms"]) for r in rows)
    y_max = y_max * 1.08 if y_max > 0 else 1.0  # 8% headroom

    configs = [(True, "1"), (True, "3"), (True, "all"), (False, "1"), (False, "3"), (False, "all")]
    for smooth, n_alt in configs:
        subset = [r for r in rows if r["smooth"] == smooth and r["n_alt"] == n_alt]
        if not subset:
            continue
        label_smooth = "smooth" if smooth else "nonsmooth"
        fig, ax = plt.subplots(figsize=(9, 5))
        fw = [next((r["forward_ms"] for r in subset if r["backend"] == b), 0.0) for b in backends]
        bw = [next((r["backward_ms"] for r in subset if r["backend"] == b), 0.0) for b in backends]
        bars1 = ax.bar(x - width / 2, fw, width, label="Forward", color="C0")
        bars2 = ax.bar(x + width / 2, bw, width, label="Backward", color="C1")
        ax.set_xticks(x)
        ax.set_xticklabels(backends, rotation=15, ha="right")
        ax.set_ylabel("Mean time per step (ms)")
        ax.set_ylim(0, y_max)
        ax.set_title(f"LUTorch profile — {label_smooth}, n_alternatives={n_alt}")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_name = f"plot_lutorch_profile_{label_smooth}_nalt{n_alt}.png"
        out_path = path.parent / out_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print("Wrote", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
