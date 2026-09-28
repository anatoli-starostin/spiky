"""Step 0: the one-hot GEMM roofline, computed and measured BEFORE any kernel is written.

Two things:
  1. the arithmetic -- full one-hot FLOPs, the waste multiplier vs our useful work, and the
     predicted slowdown against the AS baseline;
  2. the 5090's ACHIEVABLE dense tensor-core throughput, measured with plain torch.matmul at
     a large square shape, rather than taken from a spec sheet.
"""
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
N, T, K, D = 24576, 256, 256, 1024
FETCH = 1.3272
AS_MS = 1.4187                     # AS read_cells, load16=False, real cells, artifacts/ts_bench.json
SIZES = [2048, 4096, 8192]


def peak(dtype, sizes=SIZES, iters=20):
    best = 0.0
    detail = []
    for n in sizes:
        a = torch.randn(n, n, device='cuda', dtype=torch.float32).to(dtype)
        b = torch.randn(n, n, device='cuda', dtype=torch.float32).to(dtype)
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        t = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            a @ b
            e.record()
            torch.cuda.synchronize()
            t.append(s.elapsed_time(e))
        ms = statistics.median(t)
        flops = 2.0 * n ** 3 / (ms * 1e-3)
        detail.append((n, ms, flops))
        best = max(best, flops)
        del a, b
        torch.cuda.empty_cache()
    return best, detail


def int8_peak(sizes=SIZES, iters=20):
    best, detail = 0.0, []
    for n in sizes:
        a = torch.randint(-127, 128, (n, n), device='cuda', dtype=torch.int8)
        b = torch.randint(-127, 128, (n, n), device='cuda', dtype=torch.int8)
        try:
            for _ in range(5):
                torch._int_mm(a, b)
            torch.cuda.synchronize()
        except Exception as ex:
            return None, f'{type(ex).__name__}: {str(ex).splitlines()[0][:110]}'
        t = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            torch._int_mm(a, b)
            e.record()
            torch.cuda.synchronize()
            t.append(s.elapsed_time(e))
        ms = statistics.median(t)
        ops = 2.0 * n ** 3 / (ms * 1e-3)
        detail.append((n, ms, ops))
        best = max(best, ops)
        del a, b
        torch.cuda.empty_cache()
    return best, detail


def main():
    print(f'SHAPE: N={N:,} T={T} K={K} D={D}, real-cell fetch rate {FETCH} cells/table')
    useful = N * T * FETCH * D
    onehot_mac = float(N) * T * K * D
    print(f'\nARITHMETIC')
    print(f'  useful work (AS)         N*T*{FETCH}*D = {useful:.4e} MAC-equivalent lane updates')
    print(f'  full one-hot MACs        N*T*K*D       = {onehot_mac:.4e}')
    print(f'  full one-hot FLOPs       2*N*T*K*D     = {2*onehot_mac:.4e}')
    print(f'  FLOP-waste multiplier    K/{FETCH}      = {K/FETCH:.1f}x')
    print(f'  (compacted GEMM, from gemm_scoping.py: U/{FETCH} = 26.3 at Bn=32 '
          f'to 103.8 at Bn=256)')
    print(f'  => full one-hot wastes {K/FETCH/26.3:.1f}x more than the BEST compacted case '
          f'and {K/FETCH/103.8:.1f}x more than the worst')

    print(f'\nMEASURED DENSE TENSOR-CORE THROUGHPUT (torch.matmul, square, not a spec sheet)')
    res = {'shape': {'N': N, 'T': T, 'K': K, 'D': D}, 'fetch': FETCH,
           'useful_macs': useful, 'onehot_macs': onehot_mac,
           'waste_multiplier': K / FETCH, 'as_ms': AS_MS}
    peaks = {}
    for name, dt in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
        p, det = peak(dt)
        peaks[name] = p
        print(f'  {name:>5}: {p/1e12:8.1f} TFLOP/s   ' +
              '  '.join(f'{n}^3 {ms:.3f} ms' for n, ms, _ in det))
    p8, det8 = int8_peak()
    if p8 is None:
        print(f'  int8 : unavailable ({det8})')
        peaks['int8'] = None
    else:
        peaks['int8'] = p8
        print(f'  int8 : {p8/1e12:8.1f} TOP/s     ' +
              '  '.join(f'{n}^3 {ms:.3f} ms' for n, ms, _ in det8))
    res['measured_peaks'] = peaks

    print(f'\nPREDICTED ONE-HOT GEMM TIME (arithmetic only, ignoring the selection build)')
    print(f'{"dtype":>7}{"peak":>12}{"GEMM ms":>11}{"vs AS":>9}')
    for name, p in peaks.items():
        if p is None:
            continue
        ms = 2 * onehot_mac / p * 1e3
        print(f'{name:>7}{p/1e12:>10.1f}T{ms:>11.2f}{ms/AS_MS:>9.2f}x')
        res.setdefault('predicted', {})[name] = {'ms': ms, 'vs_as': ms / AS_MS}

    sel_elems = float(N) * T * K
    print(f'\nSELECTION-MATRIX COST (the thing the H100 note blamed)')
    print(f'  one-hot entries   N*T*K = {sel_elems:.3e}')
    for nm, b in (('bf16', 2), ('int8', 1)):
        print(f'  materialised {nm}: {sel_elems*b/1e9:6.2f} GB   '
              f'(write + read = {2*sel_elems*b/1e9:6.2f} GB of traffic)')
    print(f'  per table, bf16 : {sel_elems*2/T/1e6:.1f} MB  '
          f'(the H100 note quoted 402-805 MB/head at its shape)')
    print(f'  for comparison, AS reads {FETCH*T*D*N/1e9:.2f} GB of table rows in total')
    res['selection'] = {'entries': sel_elems, 'bf16_bytes': sel_elems * 2,
                        'int8_bytes': sel_elems}

    json.dump(res, open(os.path.join(HERE, 'artifacts', 'gemm_roofline.json'), 'w'), indent=1)
    print('\nwrote artifacts/gemm_roofline.json')


if __name__ == '__main__':
    main()
