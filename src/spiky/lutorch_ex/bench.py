"""A small, general benchmark harness for lutorch_ex cartridges.

Measures any :class:`~spiky.lutorch_ex.lut_base.MultiHeadLUT` cartridge over a grid of

* operation: ``forward_eval`` (no-grad eval), ``forward_train`` (grad-enabled forward),
  ``backward`` (the ``loss.backward()`` pass, timed in isolation from the forward);
* batch size: ``1, 128, 24576`` by default;
* device: ``cpu`` plus named GPU targets (``cuda:H100``, ``cuda:RTX5090``), extensible.

Methodology: warmup iterations, ``torch.cuda.synchronize()`` around CUDA timing, several
repeats reported as median and min (ms) plus throughput (rows/s), ``no_grad`` for eval and
``requires_grad`` + ``loss.backward()`` for the backward measurement. A device that is not
present (or whose measurement is disabled) yields a clear ``placeholder`` row instead of
crashing, and any per-cell failure (e.g. OOM) is caught and reported as an ``error`` row.

Usable as a library (``benchmark(...)``, the general harness) or as a demo script
(``python -m spiky.lutorch_ex.bench``, which times only ManifestoHardLUT / ManifestoSoftLUT at one small
fixed geometry).
"""
from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict, dataclass
from statistics import median
from typing import Callable, Optional, Sequence

import torch

from .lut_base import MultiHeadLUT

OPERATIONS: tuple[str, ...] = ("forward_eval", "forward_train", "backward")
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 128, 24576)
DEFAULT_DEVICES: tuple[str, ...] = ("cpu", "cuda:H100", "cuda:RTX5090")

# Map a named GPU target to a substring expected in torch.cuda.get_device_name().
_GPU_TOKENS = {"h100": "h100", "rtx5090": "5090", "5090": "5090", "a100": "a100"}


@dataclass
class BenchRow:
    cartridge: str
    device: str
    operation: str
    batch_size: int
    status: str                       # "ok" | "placeholder" | "error"
    median_ms: Optional[float] = None
    min_ms: Optional[float] = None
    throughput_rows_per_s: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_device(label: str, measure_cuda: bool) -> tuple[Optional[torch.device], str]:
    """Return (device, reason). device is None -> emit a placeholder row with `reason`."""
    if label == "cpu":
        return torch.device("cpu"), ""
    target = label.split(":", 1)[1] if ":" in label else label
    if not torch.cuda.is_available():
        return None, "CUDA not available on this host"
    name = torch.cuda.get_device_name(0)
    tok = _GPU_TOKENS.get(target.lower(), target.lower())
    if tok and tok not in name.lower().replace(" ", ""):
        return None, f"present GPU is {name!r}, not {target}"
    if not measure_cuda:
        return None, f"{name} present but CUDA benchmarking disabled (pass measure_cuda=True)"
    return torch.device("cuda:0"), ""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_make_input(
    cart: MultiHeadLUT, batch: int, device: torch.device, requires_grad: bool
) -> torch.Tensor:
    spec = cart.spec
    x = torch.randn(batch, spec.h_in, spec.d_in, device=device)
    if requires_grad:
        x.requires_grad_(True)
    return x


def _stats(times_s: Sequence[float], batch: int) -> tuple[float, float, float]:
    med = median(times_s)
    mn = min(times_s)
    thr = (batch / med) if med > 0 else float("inf")
    return med * 1e3, mn * 1e3, thr


def _run_forward(unit: Callable[[], object], device: torch.device, warmup: int, repeats: int):
    for _ in range(warmup):
        unit()
    _sync(device)
    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        unit()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def _run_backward(
    cart: MultiHeadLUT, make_input, batch: int, device: torch.device, warmup: int, repeats: int
):
    times = []
    for i in range(warmup + repeats):
        x = make_input(cart, batch, device, True)
        out = cart(x)
        loss = out.float().pow(2).sum()
        cart.zero_grad(set_to_none=True)
        _sync(device)
        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return times


def _measure_cell(
    cart: MultiHeadLUT, make_input, op: str, batch: int, device: torch.device,
    warmup: int, repeats: int,
) -> tuple[list[float], str]:
    """Return (timings_seconds, note). Raises are turned into an error note by the caller."""
    if op == "forward_eval":
        cart.eval()
        x = make_input(cart, batch, device, False)
        with torch.no_grad():
            return _run_forward(lambda: cart(x), device, warmup, repeats), ""
    if op == "forward_train":
        cart.train()
        x = make_input(cart, batch, device, True)
        return _run_forward(lambda: cart(x), device, warmup, repeats), ""
    if op == "backward":
        cart.train()
        return _run_backward(cart, make_input, batch, device, warmup, repeats), ""
    raise ValueError(f"unknown operation {op!r}")


def benchmark(
    make_cartridge: Callable[[], MultiHeadLUT],
    *,
    name: str = "cartridge",
    make_input: Optional[Callable] = None,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    operations: Sequence[str] = OPERATIONS,
    devices: Sequence[str] = DEFAULT_DEVICES,
    warmup: int = 3,
    repeats: int = 10,
    measure_cuda: bool = False,
) -> list[BenchRow]:
    """Benchmark a cartridge over the (operation x batch x device) grid. See module docstring."""
    make_input = make_input or _default_make_input
    rows: list[BenchRow] = []
    for label in devices:
        device, reason = _resolve_device(label, measure_cuda)
        if device is None:
            for op in operations:
                for batch in batch_sizes:
                    rows.append(BenchRow(name, label, op, batch, "placeholder", note=reason))
            continue
        for op in operations:
            for batch in batch_sizes:
                try:
                    cart = make_cartridge().to(device)
                    times, note = _measure_cell(cart, make_input, op, batch, device, warmup, repeats)
                    med, mn, thr = _stats(times, batch)
                    rows.append(BenchRow(name, label, op, batch, "ok", med, mn, thr, note))
                except Exception as e:  # OOM, etc. — one cell failing must not kill the grid
                    rows.append(BenchRow(name, label, op, batch, "error", note=f"{type(e).__name__}: {e}"))
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    return rows


def format_table(rows: Sequence[BenchRow]) -> str:
    """Render rows as a fixed-width text table."""
    header = ("cartridge", "device", "operation", "batch", "status", "median_ms", "min_ms", "rows/s")

    def cell(r: BenchRow) -> tuple:
        fmt = lambda v, f: (f % v) if v is not None else "-"
        return (r.cartridge, r.device, r.operation, str(r.batch_size), r.status,
                fmt(r.median_ms, "%.3f"), fmt(r.min_ms, "%.3f"), fmt(r.throughput_rows_per_s, "%.1f"))

    table = [header] + [cell(r) for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = []
    for ri, row in enumerate(table):
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(header))))
    notes = [f"  [{r.device}/{r.operation}/{r.batch_size}] {r.note}"
             for r in rows if r.note and r.status != "ok"]
    if notes:
        lines.append("")
        lines.append("notes:")
        lines.extend(sorted(set(notes)))
    return "\n".join(lines)


def rows_to_json(rows: Sequence[BenchRow]) -> str:
    return json.dumps([r.to_dict() for r in rows], indent=2)


def rows_to_csv(rows: Sequence[BenchRow]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(BenchRow.__dataclass_fields__))
    w.writeheader()
    for r in rows:
        w.writerow(r.to_dict())
    return buf.getvalue()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Demo CLI: time ManifestoHardLUT and ManifestoSoftLUT at one small fixed geometry.

    For any other cartridge or geometry, call :func:`benchmark` (the general harness) directly.
    """
    import argparse

    from .cartridges import ManifestoHardLUT, ManifestoSoftLUT
    from .lut_spec import LUTSpec

    p = argparse.ArgumentParser(
        description="Demo: time ManifestoHardLUT and ManifestoSoftLUT at one small fixed geometry "
                    "(LUTSpec(h_in=2, h_out=2, tph=8, nap=5, d_in=16, d_out=16)). For any other cartridge "
                    "or geometry, call spiky.lutorch_ex.bench.benchmark() from Python.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--measure-cuda", action="store_true",
                   help="actually time on a present matching GPU (off by default)")
    p.add_argument("--json", type=str, default=None, help="write results JSON to this path")
    p.add_argument("--csv", type=str, default=None, help="write results CSV to this path")
    args = p.parse_args(argv)

    # A modest, representative geometry so the CPU grid (incl. batch 24576) runs quickly.
    spec = LUTSpec(h_in=2, h_out=2, tph=8, nap=5, d_in=16, d_out=16)
    cartridges = [
        ("ManifestoHardLUT", lambda: ManifestoHardLUT(spec, seed=0)),
        ("ManifestoSoftLUT", lambda: ManifestoSoftLUT(spec, seed=0)),
    ]
    all_rows: list[BenchRow] = []
    for nm, factory in cartridges:
        all_rows += benchmark(factory, name=nm, warmup=args.warmup, repeats=args.repeats,
                              measure_cuda=args.measure_cuda)

    print(f"spec: {spec}")
    print(format_table(all_rows))
    if args.json:
        with open(args.json, "w") as f:
            f.write(rows_to_json(all_rows))
        print(f"\nwrote {args.json}")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(rows_to_csv(all_rows))
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
