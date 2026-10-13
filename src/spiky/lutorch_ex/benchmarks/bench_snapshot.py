"""Current performance metrics snapshot of every lutorch_ex cartridge on one CUDA device, at four geometries, in fp32
and (where the class accepts it) bf16. Four measurements per cell, each as median latency and peak memory:
eval forward, train forward, train backward, train total (forward + backward).

    python -m spiky.lutorch_ex.benchmarks.bench_snapshot --out snapshot.json --md snapshot.md

runs the full matrix (every cartridge and alternative, every geometry, fp32 and bf16), one fresh process per cell.

Methodology:
- Every cell is built with a fixed seed (``seed=1`` tables, ``torch.manual_seed(0)`` inputs). Eval forward runs in
  eval mode under ``no_grad``; train forward / backward / total run in train mode, ``y.backward(go)``.
- Each class runs with its defaults, i.e. what a user gets. A class has exactly one implementation (the class name
  says which): the fused classes their CUDA kernels (a case those do not cover is an error cell), the reference
  classes plain PyTorch, compiled on CUDA.
- Every cell runs in its own fresh Python process (``--no-isolate`` runs them all in this one).
- Per mode, the first call (build / compile) and ``--warmup`` iterations are discarded. Then ``--repeats``
  iterations are timed one by one with CUDA events, with ``torch.cuda.synchronize()`` around each, and reported as
  the median and the inter-quartile range (p25-p75). A train iteration records three events (before the forward,
  between forward and backward, after the backward), so train forward, train backward and train total come from the
  same iterations.
- Peak memory is ``max_memory_allocated`` above what was allocated before the step (inputs, parameters and their
  gradients excluded), measured on one extra iteration: eval forward; train forward (including what it saves for
  the backward); train backward (the peak while the backward runs, which still holds those saved tensors); train
  total = the larger of the two.
- A cell that runs out of memory, or that the class refuses (QuantisedConfidenceLUT rejects bf16), is recorded as
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
from spiky.lutorch_ex.lut_spec import LUTSpec

#: name -> (spec kwargs, batch of vectors). "reference" is the library's h8 / tph64 / nap8 geometry at r = h*d = 64.
GEOMETRIES = {
    "small": (dict(h_in=4, h_out=4, tph=16, nap=6, d_in=16, d_out=16), 1024),
    "reference": (dict(h_in=8, h_out=8, tph=64, nap=8, d_in=8, d_out=8), 8192),
    "champion": (dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48), 8192),
    "large": (dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48), 32768),
}

def _conf(cls, n, **kw):
    return lambda s: cls(s, seed=1, read_top_n=n, **kw)


def _fused(cls, **kw):
    return lambda s: cls(s, seed=1, **kw)


#: (name, constructor). Every concrete cartridge class (reference and fused) with its defaults.
#: DeployedQuantisedConfidenceLUT is inference-only (forward only, built from an export of a QuantisedConfidenceLUT with
#: that read width; the n=2 row keeps its earlier name).
CARTRIDGES = [
    ("ManifestoHardLUT", lambda s: lx.ManifestoHardLUT(s, seed=1)),
    ("ManifestoSoftLUT", lambda s: lx.ManifestoSoftLUT(s, seed=1)),
    ("FusedManifestoHardLUT", _fused(lx.FusedManifestoHardLUT)),
    ("FusedManifestoSoftLUT", _fused(lx.FusedManifestoSoftLUT)),
    ("ManifestoHardLUT weight=softsign", lambda s: lx.ManifestoHardLUT(s, seed=1, weight="softsign")),
    ("ManifestoSoftLUT weight=softsign", lambda s: lx.ManifestoSoftLUT(s, seed=1, weight="softsign")),
    ("FusedManifestoHardLUT weight=softsign", _fused(lx.FusedManifestoHardLUT, weight="softsign")),
    ("FusedManifestoSoftLUT weight=softsign", _fused(lx.FusedManifestoSoftLUT, weight="softsign")),
    ("ConfidenceLUT n=1", _conf(lx.ConfidenceLUT, 1)),
    ("ConfidenceLUT n=2", _conf(lx.ConfidenceLUT, 2)),
    ("FusedConfidenceLUT n=1", _conf(lx.FusedConfidenceLUT, 1)),
    ("FusedConfidenceLUT n=2", _conf(lx.FusedConfidenceLUT, 2)),
    ("QuantisedConfidenceLUT n=2", _conf(lx.QuantisedConfidenceLUT, 2)),
    ("FusedQuantisedConfidenceLUT n=2", _conf(lx.FusedQuantisedConfidenceLUT, 2)),
    ("DeployedQuantisedConfidenceLUT", lambda s: _deployed(s, 2)),       # from QuantisedConfidenceLUT.to_deployment()
    ("DeployedQuantisedConfidenceLUT n=1", lambda s: _deployed(s, 1)),
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
                                  cwd=__file__.rsplit("/", 2)[0]).stdout.strip() or "unknown"   # lutorch_ex/
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
    return _stats(ts)


def _stats(ts):
    q = statistics.quantiles(ts, n=4)
    return {"median_ms": statistics.median(ts), "p25_ms": q[0], "p75_ms": q[2]}


def _time_train(m, xx, go, warmup: int, repeats: int):
    """Train forward, train backward and train total from the same timed iterations."""
    def step(ev=None):
        if ev:
            ev[0].record()
        y = m(xx)
        if ev:
            ev[1].record()
        y.backward(go.to(y.dtype))
        if ev:
            ev[2].record()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    f, b, t = [], [], []
    for _ in range(repeats):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        torch.cuda.synchronize()
        step(ev)
        torch.cuda.synchronize()
        f.append(ev[0].elapsed_time(ev[1]))
        b.append(ev[1].elapsed_time(ev[2]))
        t.append(ev[0].elapsed_time(ev[2]))
    return _stats(f), _stats(b), _stats(t)


def _deployed(spec, n):
    q = lx.QuantisedConfidenceLUT(spec, seed=1, read_top_n=n)
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
        m = make(spec).cuda()
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
        row["eval_forward"] = _time(fwd, warmup, repeats)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fwd()
        torch.cuda.synchronize()
        row["eval_forward"]["peak_gib"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
        if isinstance(m, lx.DeployedQuantisedConfidenceLUT):   # inference-only
            return row
        m.train()
        xx = x.clone().requires_grad_(True)
        y = m(xx)                                          # first call: build / compile outside the timing
        y.backward(go.to(y.dtype))
        del y
        tf, tb, tt = _time_train(m, xx, go, warmup, repeats)
        # Peak memory: the gradients exist from here on, so they are part of the base, as in a training loop.
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y = m(xx)
        torch.cuda.synchronize()
        tf["peak_gib"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
        torch.cuda.reset_peak_memory_stats()
        y.backward(go.to(y.dtype))
        del y
        torch.cuda.synchronize()
        tb["peak_gib"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
        tt["peak_gib"] = max(tf["peak_gib"], tb["peak_gib"])
        row["train_forward"], row["train_backward"], row["train_total"] = tf, tb, tt
    except torch.OutOfMemoryError:
        row["status"] = "out of memory"
    except TypeError as e:
        row["status"] = f"refused: {str(e).split(':')[0]}"
    except Exception as e:
        row["status"] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
    finally:
        torch.cuda.empty_cache()
    return row


MEASUREMENTS = ("eval_forward", "train_forward", "train_backward", "train_total")


def markdown(env, rows, obs) -> str:
    """One line per cell: median ms and peak GiB for each of the four measurements (the IQRs are in the JSON); the
    status column says why a cell has no numbers (out of memory, refused, an error)."""
    def cell(t):
        return "- | -" if not t or "peak_gib" not in t else f"{t['median_ms']:.3f} | {t['peak_gib']:.3f}"
    head = " | ".join(f"{m.replace('_', ' ')} ms | GiB" for m in MEASUREMENTS)
    out = [f"| geometry | cartridge | dtype | status | {head} |", "|---" * (4 + 2 * len(MEASUREMENTS)) + "|"]
    for r in rows:
        out.append(f"| {r['geometry']} | {r['cartridge']} | {r['dtype']} | {r.get('status', 'ok')} | "
                   + " | ".join(cell(r.get(m)) for m in MEASUREMENTS) + " |")
    return "\n".join(out)


def _isolated(g, name, dtype, a) -> dict:
    """One cell in a fresh Python process; its row, or a status when the process itself failed."""
    import os
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cell.json")
        cmd = [sys.executable, "-m", "spiky.lutorch_ex.benchmarks.bench_snapshot", "--no-isolate", "--geometries", g,
               "--cartridges", name, "--dtypes", dtype, "--warmup", str(a.warmup), "--repeats", str(a.repeats),
               "--out", out, "--md", os.path.join(tmp, "cell.md")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(out):
            return json.load(open(out))["rows"][0]
    tail = (res.stderr.strip().splitlines() or ["no output"])[-1][:120]
    return {"cartridge": name, "dtype": dtype, "status": f"error: process exit {res.returncode}: {tail}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="snapshot.json")
    ap.add_argument("--md", default="snapshot.md")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--geometries", default=",".join(GEOMETRIES))
    ap.add_argument("--cartridges", default="", help="comma-separated cartridge names to keep (exact); default: all")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--no-isolate", dest="isolate", action="store_false",
                    help="run every cell in this process (default: one fresh process per cell)")
    a = ap.parse_args()
    env = environment()
    obs = {"start": observe()}
    rows = []
    keep = set(filter(None, a.cartridges.split(",")))
    for g in a.geometries.split(","):
        kw, B = GEOMETRIES[g]
        spec = LUTSpec(**kw)
        for name, make in CARTRIDGES:
            if keep and name not in keep:
                continue
            for d in a.dtypes.split(","):
                if a.isolate:
                    r = _isolated(g, name, d, a)
                else:
                    r = measure(name, make, spec, B, getattr(torch, d), a.warmup, a.repeats)
                r.update(geometry=g, batch=B)
                rows.append(r)
                print(json.dumps(r), flush=True)
        obs[f"after_{g}"] = observe()
    json.dump({"environment": env, "observed": obs, "rows": rows}, open(a.out, "w"), indent=1)
    open(a.md, "w").write(markdown(env, rows, obs) + "\n")


if __name__ == "__main__":
    main()
