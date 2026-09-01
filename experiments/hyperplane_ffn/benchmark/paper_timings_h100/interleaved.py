#!/usr/bin/env python3
"""Interleaved H100 FFN-slot TOTALS -- the citable numbers -- plus the torch.compile check.
H100 counterpart of ../paper_timings/interleaved.py.

    python paper_timings_h100/interleaved.py

Every variant (vanilla dense slot, and each CompressionMHL fused_v2 slot) is timed in
ALTERNATING rounds inside one process, so clock drift applies to all equally and the
RATIOS are sound; the [min-max] spread shows the residual noise. The torch.compile
variants are timed in the same interleave (compilation itself is excluded -- the warm-up
calls every variant, compiled ones included, before timing).

The routed best path is gather_fused_v2_h100.fused_v2 (routing+gather fused, index in
shared, fp32 bit-exact): compress(bf16) -> fused_v2(fp32) -> decompress(bf16).

NOTE for quoting: on H100 torch.compile helps NEITHER path -- the vanilla bf16 GEMMs are
already cuBLAS-optimal (eager ~= compiled, measured 0.156 vs 0.160 ms), and the routed
path graph-breaks at the pybind11 custom routing kernel so it is unchanged. (This differs
from the 5090, where compile gave the vanilla baseline ~1.1x.) Run on an IDLE GPU.
"""
import argparse
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)                       # experiments/hyperplane_ffn/benchmark
R = os.path.dirname(BENCH)                          # experiments/hyperplane_ffn
REPO = os.path.dirname(os.path.dirname(R))          # repo root
for p in (BENCH, os.path.join(REPO, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

import bench            # noqa: E402
import model as M       # noqa: E402

DEFAULT_EXPS = ('exp_n_0126_grid_H4d48_nap7_tph64',
                'exp_n_0127_grid_H4d48_nap7_tph128',
                'exp_n_0128_grid_H4d48_nap8_tph64')


def load_fused():
    from torch.utils.cpp_extension import load
    return load(name='hyperplane_fused_v2',
                sources=[os.path.join(BENCH, 'gather_fused_v2_h100.cu')],
                extra_cuda_cflags=['-O3', '-std=c++20', '--use_fast_math'],
                extra_cflags=['-O3', '-std=c++20'], verbose=False)


def timeit(fn, iters):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def interleave(variants, rounds, iters, warmup=20):
    with torch.no_grad():
        for _ in range(warmup):
            for fn in variants.values():
                fn()
        torch.cuda.synchronize()
        acc = {n: [] for n in variants}
        for _ in range(rounds):
            for n, fn in variants.items():
                acc[n].append(timeit(fn, iters))
    return {n: (statistics.median(v), min(v), max(v)) for n, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=R)
    ap.add_argument('--exps', default=','.join(DEFAULT_EXPS))
    ap.add_argument('--baseline', default='exp_n_0135_untied_vanilla_baseline_16k')
    ap.add_argument('--compile-exp', default='exp_n_0127_grid_H4d48_nap7_tph128',
                    help='which LUT model to also run under torch.compile')
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--rounds', type=int, default=11)
    ap.add_argument('--no-compile', action='store_true')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')
    B, T, Ch, DEV = a.batch, a.seq, 384, 'cuda'
    N = B * T
    ef = load_fused()
    exps = a.exps.split(',')
    print(f'GPU: {bench.gpu_name()} | torch {torch.__version__} | batch {B}x{T}={N} tok | '
          f'{a.iters} iters x {a.rounds} interleaved rounds | fp32 fused_v2 (bit-exact), bf16 vanilla')

    # vanilla dense slot (bf16)
    _, van = M.build(os.path.join(a.root, a.baseline))
    van.blocks[0].mlp.to(torch.bfloat16)
    xv = torch.randn(B, T, Ch, device=DEV, dtype=torch.bfloat16)
    variants = {'vanilla': (lambda: van.blocks[0].ffn_slot(xv))}

    keep = []
    for exp in exps:
        cfg, ref = M.build(os.path.join(a.root, exp), load_checkpoint=True)
        ffn = ref.blocks[0].ffn
        lut = M.lut_modules(ref)[0]
        H, tph, nap = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
        A, Bn = lut.soft_anchor_a_long, lut.soft_anchor_b_long
        W = lut.weights.data.contiguous()
        cbf = ffn.compress.to(torch.bfloat16)
        dbf = ffn.decompress.to(torch.bfloat16)
        x = torch.randn(N, Ch, device=DEV, dtype=torch.bfloat16)
        nout = lut.n_outputs
        keep.append((cbf, dbf, x, A, Bn, W, H, tph, nap, nout))

        def slot(_c=cbf, _d=dbf, _x=x, _A=A, _B=Bn, _W=W, _H=H, _t=tph, _n=nap, _o=nout):
            z = _c(_x).float().contiguous()
            y = ef.fused_v2(z, _A, _B, _W, _H, _t, _n, 64)
            return _d(y.to(torch.bfloat16).reshape(N, _H * _o))
        short = exp.split('_')[2] if exp.startswith('exp_n_') else exp
        variants[f'{short}_fused_v2'] = slot

    res = interleave(variants, a.rounds, a.iters)
    van_med = res['vanilla'][0]
    print('\nFFN-slot totals, interleaved (ms/call):')
    print(f'  {"variant":<18}{"median":>9}{"[min-max]":>18}{"vs vanilla":>12}')
    for n, (m, lo, hi) in res.items():
        ratio = '' if n == 'vanilla' else f'{m / van_med:.2f}x'
        print(f'  {n:<18}{m:>9.4f}   [{lo:.4f}-{hi:.4f}]{ratio:>12}')

    if not a.no_compile:
        print('\ntorch.compile check (same interleave):')
        cvar = {'vanilla_eager': (lambda: van.blocks[0].ffn_slot(xv)),
                'vanilla_compiled': None}
        cvan = torch.compile(lambda z: van.blocks[0].ffn_slot(z))
        cvar['vanilla_compiled'] = (lambda: cvan(xv))
        cfg, cref = M.build(os.path.join(a.root, a.compile_exp), load_checkpoint=True)
        cblk = cref.blocks[0]
        hh = torch.randn(B, T, Ch, device=DEV)
        crt = torch.compile(lambda z: cblk.ffn_slot(z))
        cvar['routed_eager(fp32)'] = (lambda: cblk.ffn_slot(hh))
        cvar['routed_compiled'] = (lambda: crt(hh))
        cres = interleave(cvar, a.rounds, a.iters)
        for n, (m, lo, hi) in cres.items():
            print(f'  {n:<20}{m:>9.4f}   [{lo:.4f}-{hi:.4f}]')
        print('  (on H100 compile helps NEITHER: vanilla GEMMs already cuBLAS-optimal; '
              'routed graph-breaks at the custom kernel.)')


if __name__ == '__main__':
    main()
