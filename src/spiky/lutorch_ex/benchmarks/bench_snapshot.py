"""Current performance metrics snapshot of every lutorch_ex cartridge on one CUDA device, at four geometries, in
float32 and (where the class accepts it) bfloat16. Four measurements per cell, each as median latency and peak memory:
eval forward, train forward, train backward, train total (forward + backward); eval forward and train total also as
GPU kernel time.

    python -m spiky.lutorch_ex.benchmarks.bench_snapshot --out snapshot.json --md snapshot.md

runs the full matrix (every cartridge and alternative, every geometry, float32 and bfloat16), one fresh process per
cell.

Methodology:
- Every cell is built with a fixed seed (``seed=1`` tables, ``torch.manual_seed(0)`` inputs). Eval forward runs in
  eval mode under ``no_grad``; train forward / backward / total run in train mode, ``y.backward(go)``.
- Each class runs with its defaults, i.e. what a user gets. A class has exactly one implementation (the class name
  says which): the fused classes their CUDA kernels (a case those do not cover is an error cell), the reference
  classes plain PyTorch, compiled on CUDA.
- Every cell runs in its own fresh Python process (``--no-isolate`` runs them all in this one). ``--rounds R`` runs
  the whole matrix R times, the cell order rotated by one each round, and reports per cell the median of the R round
  medians with their range (``round_min_ms`` / ``round_max_ms``); the per-round rows are kept in the JSON.
- Per mode, the first call (build / compile) and ``--warmup`` iterations are discarded. Then ``--repeats``
  iterations are timed one by one with CUDA events, with ``torch.cuda.synchronize()`` around each, and reported as
  the median and the inter-quartile range (p25-p75). Before each timed iteration a 256 MiB buffer is overwritten
  (outside the timing) so the L2 cache is cold, as it is in a training step. A train iteration records three events
  (before the forward, between forward and backward, after the backward), so train forward, train backward and
  train total come from the same iterations.
- Wall time from CUDA events includes the CPU's dispatch (Python, Dynamo guards, autograd); a cell whose kernels are
  short is dispatch-bound, and its wall time varies between processes. So eval forward and train total also report
  ``gpu_ms``: the summed kernel time of one step (torch.profiler, averaged over 10 steps, the L2 flush excluded),
  free of that dispatch. The profiler itself slows a kernel-dense step (a compiled train step: ~10-20%), so a
  ``gpu_ms`` above 1.05 x the wall median is flagged ``gpu_suspect``; there the wall time is the better number.
- Peak memory is ``max_memory_allocated`` above what was allocated before the step (inputs, parameters and their
  gradients excluded), measured on one extra iteration: eval forward; train forward (including what it saves for
  the backward); train backward (the peak while the backward runs, which still holds those saved tensors); train
  total = the larger of the two.
- A cell that runs out of memory, or that the class refuses (QuantisedConfidenceLUT rejects bfloat16), is recorded as
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
    ("QuantisedConfidenceLUT n=1", _conf(lx.QuantisedConfidenceLUT, 1)),
    ("QuantisedConfidenceLUT n=2", _conf(lx.QuantisedConfidenceLUT, 2)),
    ("FusedQuantisedConfidenceLUT n=1", _conf(lx.FusedQuantisedConfidenceLUT, 1)),
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


#: Overwritten before every timed iteration so the L2 cache is cold (allocated once per process, outside the peaks).
_FLUSH_BYTES = 256 * 2 ** 20
_flush_buf = None


def _flush_l2() -> None:
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    _flush_buf.zero_()


def _gpu_ms(fn, steps: int = 10) -> float:
    """Summed CUDA kernel time of one call of ``fn`` (ms), averaged over ``steps`` calls, each after an L2 flush whose
    kernel is not counted. The first profiler session of a process records no CUDA events, so a throwaway session
    runs first."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts):
        fn()
        torch.cuda.synchronize()
    with profile(activities=acts) as prof:
        for _ in range(steps):
            _flush_l2()
            fn()
        torch.cuda.synchronize()
    # Kernel rows only: device type CUDA (CPU-side rows -- aten ops, a custom autograd.Function -- carry their kernels'
    # time again), and not a GPU range: Inductor records each compiled graph as a CUDA-typed "## Call CompiledFxGraph"
    # range spanning its kernels.
    # CUPTI's own bookkeeping ("Activity Buffer Request", "Lazy Function Loading") can appear with a device time too.
    total_us = sum(e.self_device_time_total for e in prof.key_averages()
                   if e.device_type == torch.autograd.DeviceType.CUDA
                   and not e.key.startswith(("## ", "Activity Buffer", "Lazy Function"))
                   and "FillFunctor<unsigned char>" not in e.key)
    return total_us / steps / 1000.0


def _check_gpu(t: dict) -> dict:
    """Kernel time cannot exceed the wall time around it: flag a profile that says otherwise (``gpu_suspect``)."""
    if "gpu_ms" in t and t["gpu_ms"] > 1.05 * t["median_ms"]:
        t["gpu_suspect"] = True
    return t


def _time(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        _flush_l2()
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
        _flush_l2()
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
        row["eval_forward"]["gpu_ms"] = _gpu_ms(fwd)
        _check_gpu(row["eval_forward"])
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

        gy = go.to(x.dtype)                                # every class returns its input's dtype

        def train_step():
            m(xx).backward(gy)
        tt["gpu_ms"] = _gpu_ms(train_step)
        _check_gpu(tt)
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


def aggregate(round_rows) -> list:
    """One row per cell (geometry, cartridge, dtype) from its rounds: per measurement the median of the round medians
    (``median_ms``) and their range (``round_min_ms`` / ``round_max_ms``), the median of the rounds' p25 / p75, GPU time
    and peak. A round that failed is reported in ``status`` with the count of rounds it failed in."""
    cells = {}
    for r in round_rows:
        cells.setdefault((r["geometry"], r["cartridge"], r["dtype"]), []).append(r)
    rows = []
    for (g, name, dt), rs in cells.items():
        row = {"geometry": g, "cartridge": name, "dtype": dt, "batch": rs[0].get("batch"), "rounds": len(rs)}
        failed = [r["status"] for r in rs if "status" in r]
        if failed:
            row["status"] = (failed[0] if len(failed) == len(rs)
                             else f"{failed[0]} (in {len(failed)} of {len(rs)} rounds)")
        for m in MEASUREMENTS:
            ts = [r[m] for r in rs if m in r]
            if not ts:
                continue
            med = [t["median_ms"] for t in ts]
            agg = {"median_ms": statistics.median(med), "round_min_ms": min(med), "round_max_ms": max(med),
                   "p25_ms": statistics.median(t["p25_ms"] for t in ts),
                   "p75_ms": statistics.median(t["p75_ms"] for t in ts)}
            for k in ("peak_gib", "gpu_ms"):
                vals = [t[k] for t in ts if k in t]
                if vals:
                    agg[k] = statistics.median(vals)
            row[m] = agg
        rows.append(row)
    return rows


def markdown(env, rows, obs) -> str:
    """One line per cell: eval forward and train total as wall median (range of the round medians), GPU kernel time
    and peak GiB, then train forward and train backward wall medians (IQRs and per-round rows are in the JSON); the
    status column says why a cell has no numbers (out of memory, refused, an error)."""
    def wall(t):
        if not t:
            return "-"
        rng = (f" ({t['round_min_ms']:.3f}-{t['round_max_ms']:.3f})"
               if "round_min_ms" in t and t["round_max_ms"] > t["round_min_ms"] else "")
        return f"{t['median_ms']:.3f}{rng}"

    def num(t, k, fmt, scale=1.0):
        return format(t[k] * scale, fmt) if t and k in t else "-"
    out = [f"{env['gpu']} ({env['sm']}), torch {env['torch']}, CUDA {env['cuda_runtime']}, driver {env['driver']}, "
           f"commit {env['commit'][:10]}{' + uncommitted' if env['uncommitted_changes_in_lutorch_ex'] else ''}", "",
           "| geometry | cartridge | dtype | status | eval wall ms | eval GPU ms | eval peak MiB | train wall ms | "
           "train GPU ms | train peak MiB | train fwd ms | train bwd ms |", "|---" * 12 + "|"]
    for r in rows:
        ev, tt = r.get("eval_forward"), r.get("train_total")
        out.append(f"| {r['geometry']} | {r['cartridge']} | {r['dtype']} | {r.get('status', 'ok')} | {wall(ev)} | "
                   f"{num(ev, 'gpu_ms', '.3f')} | {num(ev, 'peak_gib', '.1f', 1024)} | {wall(tt)} | "
                   f"{num(tt, 'gpu_ms', '.3f')} | {num(tt, 'peak_gib', '.1f', 1024)} | "
                   f"{num(r.get('train_forward'), 'median_ms', '.3f')} | "
                   f"{num(r.get('train_backward'), 'median_ms', '.3f')} |")
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
            return json.load(open(out))["round_rows"][0]
    tail = (res.stderr.strip().splitlines() or ["no output"])[-1][:120]
    return {"cartridge": name, "dtype": dtype, "status": f"error: process exit {res.returncode}: {tail}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="snapshot.json")
    ap.add_argument("--md", default="snapshot.md")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=1,
                    help="run the matrix this many times, the cell order rotated by one each round")
    ap.add_argument("--geometries", default=",".join(GEOMETRIES))
    ap.add_argument("--cartridges", default="", help="comma-separated cartridge names to keep (exact); default: all")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--no-isolate", dest="isolate", action="store_false",
                    help="run every cell in this process (default: one fresh process per cell)")
    a = ap.parse_args()
    env = environment()
    obs = {"start": observe()}
    keep = set(filter(None, a.cartridges.split(",")))
    cells = [(g, name, make, d) for g in a.geometries.split(",") for name, make in CARTRIDGES
             if not keep or name in keep for d in a.dtypes.split(",")]
    round_rows = []
    for rnd in range(a.rounds):
        k = rnd % len(cells)
        for g, name, make, d in cells[k:] + cells[:k]:
            kw, B = GEOMETRIES[g]
            if a.isolate:
                r = _isolated(g, name, d, a)
            else:
                r = measure(name, make, LUTSpec(**kw), B, getattr(torch, d), a.warmup, a.repeats)
            r.update(geometry=g, batch=B, round=rnd)
            round_rows.append(r)
            print(json.dumps(r), flush=True)
        obs[f"after_round_{rnd}"] = observe()
    rows = aggregate(round_rows)
    json.dump({"environment": env, "observed": obs, "settings": vars(a), "rows": rows, "round_rows": round_rows},
              open(a.out, "w"), indent=1)
    open(a.md, "w").write(markdown(env, rows, obs) + "\n")


if __name__ == "__main__":
    main()
