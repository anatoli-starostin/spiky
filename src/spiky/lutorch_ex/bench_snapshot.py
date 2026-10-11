"""Current performance metrics snapshot of every lutorch_ex cartridge on one CUDA device: forward latency,
forward+backward latency and peak memory, at four geometries, in fp32 and (where the class accepts it) bf16.

    python -m spiky.lutorch_ex.bench_snapshot --out snapshot.json --md snapshot.md

Methodology:
- Every cell is built with a fixed seed (``seed=1`` tables, ``torch.manual_seed(0)`` inputs), in train mode for
  forward+backward and in eval mode under ``no_grad`` for the forward.
- ``backend='auto'``, i.e. what a user gets; the backend that actually ran is recorded.
- Per measurement, ``--warmup`` iterations are discarded. Then ``--repeats`` iterations are timed one by one with
  CUDA events, with ``torch.cuda.synchronize()`` around each, and reported as the median and the inter-quartile
  range (p25-p75).
- Peak memory is ``max_memory_allocated`` during one extra forward+backward, above what was allocated before it
  (inputs and parameters excluded).
- A cell that runs out of memory, or that the class refuses (the reference cartridges reject bf16), is recorded as
  such, never dropped.
- ``torch._dynamo.reset()`` before every cell: all cartridges of a class compile the same ``_forward_impl``, so without
  it the cells after the first eight compiles hit Dynamo's recompile limit and silently measure eager execution.
- The environment (commit, torch / CUDA / driver, GPU and VRAM, observed SM clock, power limit and draw,
  temperature, other processes on the card) is captured at the start and after each geometry, so a later run can
  be compared like for like. Clocks are not locked (that needs root); the observed range is recorded instead.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import torch
import torch._dynamo

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges.fused_confidence import FusedConfidenceLUT
from spiky.lutorch_ex.lut_spec import LUTSpec

#: name -> (spec kwargs, batch of vectors). "reference" is the library's h8 / tph64 / nap8 geometry at r = h*d = 64.
GEOMETRIES = {
    "small": (dict(h_in=4, h_out=4, tph=16, nap=6, d_in=16, d_out=16), 1024),
    "reference": (dict(h_in=8, h_out=8, tph=64, nap=8, d_in=8, d_out=8), 8192),
    "champion": (dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48), 8192),
    "large": (dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48), 32768),
}

#: Every concrete cartridge class, read from the package (reference and fused), with the options that select a
#: distinct read. DeployedQuantisedConfidenceLUT is inference-only (forward only, built from an export).
CARTRIDGES = [
    ("ManifestoHardLUT", lambda s: lx.ManifestoHardLUT(s, seed=1)),
    ("ManifestoSoftLUT", lambda s: lx.ManifestoSoftLUT(s, seed=1)),
    ("FusedManifestoHardLUT", lambda s: lx.FusedManifestoHardLUT(s, seed=1)),
    ("FusedManifestoSoftLUT", lambda s: lx.FusedManifestoSoftLUT(s, seed=1)),
    ("SoftSignHardLUT", lambda s: lx.SoftSignHardLUT(s, seed=1)),
    ("SoftSignSmoothLUT", lambda s: lx.SoftSignSmoothLUT(s, seed=1)),
    ("FusedSoftSignHardLUT", lambda s: lx.FusedSoftSignHardLUT(s, seed=1)),
    ("FusedSoftSignSmoothLUT", lambda s: lx.FusedSoftSignSmoothLUT(s, seed=1)),
    ("ConfidenceLUT n=1", lambda s: lx.ConfidenceLUT(s, seed=1, read_top_n=1)),
    ("ConfidenceLUT n=2", lambda s: lx.ConfidenceLUT(s, seed=1, read_top_n=2)),
    ("FusedConfidenceLUT n=1", lambda s: FusedConfidenceLUT(s, seed=1, read_top_n=1)),
    ("FusedConfidenceLUT n=2", lambda s: FusedConfidenceLUT(s, seed=1, read_top_n=2)),
    ("QuantisedConfidenceLUT n=2", lambda s: lx.QuantisedConfidenceLUT(s, seed=1, read_top_n=2)),
    ("DeployedQuantisedConfidenceLUT", None),            # built from QuantisedConfidenceLUT.to_deployment()
]


def _smi(query: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception as e:                                   # no nvidia-smi: say so
        return f"unavailable ({type(e).__name__})"


def _apps() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "none"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def environment() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10,
                                  cwd=__file__.rsplit("/", 1)[0]).stdout.strip() or "unknown"
        except Exception:
            return "unknown"
    p = torch.cuda.get_device_properties(0)
    return {
        "commit": git("rev-parse", "HEAD"),
        "uncommitted_changes_in_lutorch_ex": git("status", "--porcelain", "--", ".") not in ("", "unknown"),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "driver": _smi("driver_version"), "gpu": p.name, "vram_gib": round(p.total_memory / 2 ** 30, 1),
        "sm": f"sm_{p.major}{p.minor}", "power_limit_w": _smi("power.limit"), "max_sm_clock": _smi("clocks.max.sm"),
        "clocks_locked": False, "other_processes": _apps(), "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def observe() -> dict:
    return {"sm_clock": _smi("clocks.sm"), "power_draw": _smi("power.draw"), "temp_c": _smi("temperature.gpu"),
            "throttle": _smi("clocks_throttle_reasons.active")}


def _time(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    q = statistics.quantiles(ts, n=4)
    return {"median_ms": statistics.median(ts), "p25_ms": q[0], "p75_ms": q[2]}


def _backend(m, x) -> str:
    """What ran: the fused cartridges that record it (last_backend); else the class's own dispatch (FusedSoftSign*
    _pick); else the class's single implementation."""
    be = getattr(m, "last_backend", None)
    if be:
        return be
    if hasattr(m, "_pick") and getattr(m, "backend", None) == "auto":
        return m._pick(x)
    return "reference"


def _deployed(spec):
    q = lx.QuantisedConfidenceLUT(spec, seed=1, read_top_n=2)
    payload = q.to_deployment()
    return lx.DeployedQuantisedConfidenceLUT(spec, payload["tensors"], payload["meta"])


def measure(name, make, spec, B, dtype, warmup, repeats) -> dict:
    row = {"cartridge": name, "dtype": str(dtype).replace("torch.", ""), "batch": B}
    # Every cartridge of a class compiles the same _forward_impl code; across geometries and train / eval that exceeds
    # Dynamo's recompile limit (8) within one process and later cells would silently run EAGER. Each cell starts from
    # an empty compile cache, as a fresh process would.
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    try:
        m = (_deployed(spec) if make is None else make(spec)).cuda()
        if dtype != torch.float32:
            m = m.to(dtype)
        torch.manual_seed(0)
        x = torch.randn(B, spec.h_in, spec.d_in, device="cuda").to(dtype)
        go = torch.randn(B, spec.h_out, spec.d_out, device="cuda").to(dtype)
        m.eval()

        def fwd():
            with torch.no_grad():
                return m(x)
        fwd()                                              # first call: build / compile outside the timing
        row["forward"] = _time(fwd, warmup, repeats)
        row["backend_eval"] = _backend(m, x)
        if make is None:                                   # inference-only
            row["forward_backward"] = None
            row["peak_gib"] = None
            return row
        m.train()
        xx = x.clone().requires_grad_(True)

        def fwd_bwd():
            y = m(xx)
            y.backward(go.to(y.dtype))
        fwd_bwd()
        row["forward_backward"] = _time(fwd_bwd, warmup, repeats)
        row["backend_train"] = _backend(m, x)
        torch.cuda.synchronize()
        m.zero_grad(set_to_none=True)
        xx.grad = None
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fwd_bwd()
        torch.cuda.synchronize()
        row["peak_gib"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    except torch.OutOfMemoryError:
        row["status"] = "out of memory"
    except TypeError as e:
        row["status"] = f"refused: {str(e).split(':')[0]}"
    except Exception as e:
        row["status"] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
    finally:
        torch.cuda.empty_cache()
    return row


def markdown(env, rows, obs) -> str:
    def cell(t):
        return "-" if not t else f"{t['median_ms']:.3f} ({t['p25_ms']:.3f}-{t['p75_ms']:.3f})"
    out = ["| geometry | cartridge | dtype | backend (train / eval) | forward ms, median (IQR) | fwd+bwd ms, median (IQR) "
           "| peak GiB |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "status" in r:
            out.append(f"| {r['geometry']} | {r['cartridge']} | {r['dtype']} | - | {r['status']} | | |")
            continue
        be = f"{r.get('backend_train', '') or '-'} / {r.get('backend_eval', '') or '-'}"
        pk = "-" if r.get("peak_gib") is None else f"{r['peak_gib']:.3f}"
        out.append(f"| {r['geometry']} | {r['cartridge']} | {r['dtype']} | {be} | {cell(r['forward'])} | "
                   f"{cell(r.get('forward_backward'))} | {pk} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="snapshot.json")
    ap.add_argument("--md", default="snapshot.md")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--geometries", default=",".join(GEOMETRIES))
    a = ap.parse_args()
    env = environment()
    obs = {"start": observe()}
    rows = []
    for g in a.geometries.split(","):
        kw, B = GEOMETRIES[g]
        spec = LUTSpec(**kw)
        for name, make in CARTRIDGES:
            for dtype in (torch.float32, torch.bfloat16):
                r = measure(name, make, spec, B, dtype, a.warmup, a.repeats)
                r["geometry"] = g
                rows.append(r)
                print(json.dumps(r), flush=True)
        obs[f"after_{g}"] = observe()
    json.dump({"environment": env, "observed": obs, "rows": rows}, open(a.out, "w"), indent=1)
    open(a.md, "w").write(markdown(env, rows, obs) + "\n")


if __name__ == "__main__":
    main()
