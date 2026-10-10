"""Benchmark ConfidenceLUTCuda against the compiled ConfidenceLUT (default and fused read), cartridge only, train
fwd+bwd, at the d24 LUT-FFN geometry (h 16, d 48, tph 64, nap 8, table dropout 0.2). Also reports each path's
error against an fp64 eager reference (same keep mask), and optionally sweeps the CUDA launch knobs.

    python -m spiky.lutorch_ex.bench_confidence_cuda [--tokens 32768] [--n 1 2] [--sweep]
"""
from __future__ import annotations

import argparse
import itertools
import statistics

import torch

from spiky.lutorch_ex.cartridges.confidence import ConfidenceLUT
from spiky.lutorch_ex.cartridges.confidence_cuda import ConfidenceLUTCuda, CudaKnobs
from spiky.lutorch_ex.lut_spec import LUTSpec

SPEC = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
KW = dict(seed=1, table_dropout_rate=0.2)


def build(kind, n, knobs=None):
    spec = LUTSpec(**SPEC)
    if kind == "cuda":
        return ConfidenceLUTCuda(spec, read_top_n=n, knobs=knobs, **KW).cuda().train()
    return ConfidenceLUT(spec, read_top_n=n, fused_read=(kind == "compiled-fused"), index_dtype=torch.int32,
                         **KW).cuda().train()


def time_step(mod, x, go, iters=20, warmup=5, reps=3):
    def step():
        y = mod(x)
        y.backward(go)
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


def errors_vs_fp64(n, B=4096):
    """Rel-norm error of the output, grad x and every parameter gradient, against fp64 eager, same keep mask."""
    torch.manual_seed(0)
    x = torch.randn(B, 16, 48, device="cuda")
    go = torch.randn(B, 16, 48, device="cuda")
    keep = torch.rand(B, 16, 64, device="cuda") < 0.8
    ref = build("compiled", n).double()
    ref._table_dropout_mask = lambda B_, dev, dt: keep.to(dt) / 0.8
    xr = x.double().requires_grad_(True)
    yr = ref._forward_impl(xr)
    yr.backward(go.double())
    want = {"y": yr.detach(), "gx": xr.grad, **{k: p.grad for k, p in ref.named_parameters()}}
    res = {}
    for kind in ("compiled", "compiled-fused", "cuda"):
        mod = build(kind, n)
        if kind == "cuda":
            mod._keep_flags = lambda B_, dev: keep
        else:
            # compiled modules draw the mask inside the graph; patch before the first (tracing) call
            mod._table_dropout_mask = lambda B_, dev, dt: keep.to(dt) / 0.8
        xi = x.clone().requires_grad_(True)
        y = mod(xi)
        y.backward(go)
        got = {"y": y.detach(), "gx": xi.grad, **{k: p.grad for k, p in mod.named_parameters()}}
        res[kind] = {k: ((got[k].double() - want[k]).norm() / want[k].norm()).item() for k in want}
        torch._dynamo.reset()
    return res


L2_CONFIGS = [  # (l2_hint, l2_window, l2_window_target)
    (0, 0.0, 0), (1, 0.0, 0), (2, 0.0, 0),
    (0, 0.5, 0), (0, 0.75, 0), (0, 1.0, 0),
    (0, 0.5, 1), (0, 1.0, 1),
    (1, 0.75, 0), (1, 1.0, 0),
]


def l2_sweep(n, x, go, rounds=3):
    """CUDA twin at the default knobs under each L2 configuration; configs interleaved, median over rounds."""
    from spiky.lutorch_ex.cartridges.confidence_cuda import confidence_cuda_ext
    ext = confidence_cuda_ext()
    l2, persist, win = ext.l2_info()
    print(f"  L2 {l2 / 2**20:.1f} MiB, max persisting {persist / 2**20:.1f} MiB, max window {win / 2**20:.1f} MiB; "
          f"W {build('cuda', n).weights.numel() * 4 / 2**20:.1f} MiB")
    times = {c: [] for c in L2_CONFIGS}
    for _ in range(rounds):
        for c in L2_CONFIGS:
            if c[1] > 0 and (persist <= 0 or win <= 0):
                continue
            k = CudaKnobs(l2_hint=c[0], l2_window=c[1], l2_window_target=c[2])
            ext.l2_reset()        # a window leaves persisting lines + the carve-out behind: start every config clean
            times[c].append(time_step(build("cuda", n, k), x, go)[0])
    ext.l2_reset()
    base = statistics.median(times[(0, 0.0, 0)])
    for c, ts in times.items():
        if ts:
            m = statistics.median(ts)
            print(f"    hint={c[0]} window={c[1]:.2f} target={'W' if c[2] == 0 else 'gW'}  {m:8.3f} ms  "
                  f"({100 * (m / base - 1):+.1f}%)  [{', '.join(f'{t:.3f}' for t in ts)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--l2-sweep", action="store_true", help="only the CUDA twin, across the L2 knobs")
    a = ap.parse_args()
    print(f"device {torch.cuda.get_device_name()}  torch {torch.__version__}  tokens {a.tokens}")
    if a.l2_sweep:
        x = torch.randn(a.tokens, 16, 48, device="cuda", requires_grad=True)
        go = torch.randn(a.tokens, 16, 48, device="cuda")
        for n in a.n:
            print(f"\n== read_top_n={n}: CUDA twin, L2 knobs (median of 3 interleaved rounds x 3 x 20 iters)")
            l2_sweep(n, x, go)
        return
    x = torch.randn(a.tokens, 16, 48, device="cuda", requires_grad=True)
    go = torch.randn(a.tokens, 16, 48, device="cuda")
    for n in a.n:
        print(f"\n== read_top_n={n}: train fwd+bwd, cartridge only (median of 3 x 20 iters), peak GiB above inputs")
        for kind in ("compiled", "compiled-fused", "cuda"):
            ms, peak = time_step(build(kind, n), x, go)
            print(f"  {kind:15s} {ms:8.3f} ms  {peak:6.2f} GiB")
            torch._dynamo.reset()
        if a.sweep:
            print("  sweep (fwd_threads, bwd_threads, rows_per_cta, vec, vec_atomics):")
            rows = []
            for ft, bt, r, v, va in itertools.product((64, 128, 256), (64, 128, 256), (1, 2, 4), (4, 2), (True,)):
                k = CudaKnobs(fwd_threads=ft, bwd_threads=bt, rows_per_cta=r, vec=v, vec_atomics=va)
                ms, _ = time_step(build("cuda", n, k), x, go, iters=10, reps=1)
                rows.append((ms, k))
            for ms, k in sorted(rows, key=lambda t: t[0])[:8]:
                print(f"    {ms:8.3f} ms  {k}")
        print("  rel-norm error vs fp64 eager (B=4096):")
        for kind, e in errors_vs_fp64(n).items():
            print(f"    {kind:15s} " + "  ".join(f"{k}={v:.1e}" for k, v in e.items()))


if __name__ == "__main__":
    main()
