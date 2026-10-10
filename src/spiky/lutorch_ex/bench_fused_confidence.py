"""Benchmark FusedConfidenceLUT against the compiled ConfidenceLUT (default and fused read), cartridge only, train
forward and fwd+bwd, at the d24 LUT-FFN geometry (h 16, d 48, tph 64, nap 8, table dropout 0.2), with an fp32 and a
bf16 table. Also reports each path's error against an fp64 eager reference (same keep mask), and optionally sweeps
the CUDA launch knobs (--sweep: fp32 table; --sweep-bf16: bf16 table).

    python -m spiky.lutorch_ex.bench_fused_confidence [--tokens 32768] [--n 1 2] [--sweep] [--sweep-bf16]

The input is fp32 in every row, so the bf16 row differs from the fp32 row only in the table (and the scalar
parameters) being stored in bf16.
"""
from __future__ import annotations

import argparse
import itertools
import os
import statistics

import torch

# A timing run must never silently measure a slow fallback: an involuntary fallback is a hard error here
# (cartridges/_fallback.py), unless the caller explicitly set the variable.
os.environ.setdefault("SPIKY_LUTORCH_REQUIRE_NATIVE", "1")

from spiky.lutorch_ex.cartridges._fallback import backend_of  # noqa: E402
from spiky.lutorch_ex.cartridges.confidence import ConfidenceLUT  # noqa: E402
from spiky.lutorch_ex.cartridges.fused_confidence import FusedConfidenceLUT, CudaKnobs  # noqa: E402
from spiky.lutorch_ex.lut_spec import LUTSpec  # noqa: E402


def provenance(mod) -> str:
    """The backend that actually ran, for the row label; raises if it cannot be determined (never report blind)."""
    if isinstance(mod, FusedConfidenceLUT):
        return backend_of(mod)
    return "compiled ConfidenceLUT, " + ("fused read" if mod.fused_read else "default read")

SPEC = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
KW = dict(seed=1, table_dropout_rate=0.2)
KINDS = ("compiled", "compiled-fused", "cuda", "cuda-bf16")


def build(kind, n, knobs=None):
    spec = LUTSpec(**SPEC)
    if kind in ("cuda", "cuda-bf16"):
        m = FusedConfidenceLUT(spec, read_top_n=n, backend="cuda", knobs=knobs, **KW).cuda().train()
        return m.to(torch.bfloat16) if kind == "cuda-bf16" else m
    return ConfidenceLUT(spec, read_top_n=n, fused_read=(kind == "compiled-fused"), index_dtype=torch.int32,
                         **KW).cuda().train()


def time_step(mod, x, go, iters=20, warmup=5, reps=3, fwd_only=False):
    def step():
        y = mod(x)
        if not fwd_only:
            y.backward(go.to(y.dtype))
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    meds = []
    for _ in range(reps):
        ts = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            step()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        meds.append(statistics.median(ts))
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    mod.zero_grad(set_to_none=True)
    return statistics.median(meds), peak


def _fp64_ref(n, x, go, keep, state=None):
    ref = build("compiled", n).double()
    if state is not None:                                  # the bf16 model's (rounded) parameters, upcast exactly
        ref.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in state.items()}, strict=False)
    ref._table_dropout_mask = lambda B_, dev, dt: keep.to(dt) / 0.8
    xr = x.double().requires_grad_(True)
    yr = ref._forward_impl(xr)
    yr.backward(go.double())
    return {"y": yr.detach(), "gx": xr.grad, **{k: p.grad for k, p in ref.named_parameters()}}


def errors_vs_fp64(n, B=4096):
    """Rel-norm error of the output, grad x and every parameter gradient against fp64 eager, same keep mask. For the
    bf16 table, against fp64 on the SAME bf16-rounded parameters (the arithmetic error) and against fp64 on the
    original fp32 parameters (arithmetic + bf16 storage)."""
    torch.manual_seed(0)
    x = torch.randn(B, 16, 48, device="cuda")
    go = torch.randn(B, 16, 48, device="cuda")
    keep = torch.rand(B, 16, 64, device="cuda") < 0.8
    want = _fp64_ref(n, x, go, keep)
    res = {}
    for kind in KINDS:
        mod = build(kind, n)
        if kind.startswith("cuda"):
            mod._keep_flags = lambda B_, dev: keep
        else:
            # compiled modules draw the mask inside the graph; patch before the first (tracing) call
            mod._table_dropout_mask = lambda B_, dev, dt: keep.to(dt) / 0.8
        xi = x.clone().requires_grad_(True)
        y = mod(xi)
        y.backward(go)
        got = {"y": y.detach(), "gx": xi.grad, **{k: p.grad for k, p in mod.named_parameters()}}
        refs = {"": want}
        if kind == "cuda-bf16":
            refs = {" (same bf16 params)": _fp64_ref(n, x, go, keep, mod.state_dict()), " (fp32 params)": want}
        for tag, w in refs.items():
            res[kind + tag] = {k: ((got[k].double() - w[k]).norm() / w[k].norm()).item() for k in w}
        torch._dynamo.reset()
    return res


def sweep(kind, n, x, go):
    rows = []
    if kind == "cuda":
        grid = itertools.product((64, 128, 256), (64, 128, 256), (1, 2, 4), (4, 2))
        mk = lambda ft, bt, r, v: CudaKnobs(fwd_threads=ft, bwd_threads=bt, rows_per_cta=r, vec=v)
    else:
        grid = itertools.product((32, 64, 128), (32, 64, 128), (2, 4), (8, 4, 2))
        mk = lambda ft, bt, r, v: CudaKnobs(fwd_threads=ft, bwd_threads=bt, rows_per_cta=r, vec_bf16=v)
    for ft, bt, r, v in grid:
        k = mk(ft, bt, r, v)
        mod = build(kind, n, k)
        ms, _ = time_step(mod, x, go, iters=10, reps=1)
        if provenance(mod) != "cuda":
            raise RuntimeError(f"sweep row ran backend {provenance(mod)!r}, not 'cuda'; refusing to report")
        rows.append((ms, k))
    for ms, k in sorted(rows, key=lambda t: t[0])[:8]:
        print(f"    {ms:8.3f} ms  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sweep", action="store_true", help="sweep the launch knobs for an fp32 table")
    ap.add_argument("--sweep-bf16", action="store_true", help="sweep the launch knobs (incl. vec_bf16) for a bf16 table")
    a = ap.parse_args()
    print(f"device {torch.cuda.get_device_name()}  torch {torch.__version__}  tokens {a.tokens}")
    x = torch.randn(a.tokens, 16, 48, device="cuda", requires_grad=True)
    go = torch.randn(a.tokens, 16, 48, device="cuda")
    for n in a.n:
        print(f"\n== read_top_n={n}: train forward / fwd+bwd, cartridge only (median of 3 x 20 iters), "
              f"peak GiB above inputs (fwd+bwd)")
        for kind in KINDS:
            mod = build(kind, n)
            fwd, _ = time_step(mod, x, go, fwd_only=True)
            ms, peak = time_step(mod, x, go)
            print(f"  {kind:15s} fwd {fwd:7.3f} ms   fwd+bwd {ms:7.3f} ms  {peak:6.2f} GiB   "
                  f"[backend: {provenance(mod)}]")
            torch._dynamo.reset()
        if a.sweep:
            print("  sweep, fp32 table (fwd_threads, bwd_threads, rows_per_cta, vec):")
            sweep("cuda", n, x, go)
        if a.sweep_bf16:
            print("  sweep, bf16 table (fwd_threads, bwd_threads, rows_per_cta, vec_bf16):")
            sweep("cuda-bf16", n, x, go)
        print("  rel-norm error vs fp64 eager (B=4096):")
        for kind, e in errors_vs_fp64(n).items():
            print(f"    {kind:34s} " + "  ".join(f"{k}={v:.1e}" for k, v in e.items()))


if __name__ == "__main__":
    main()
