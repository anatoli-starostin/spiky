"""Variant (b): the whole one-hot read as ONE matmul, [N, T*K] @ [T*K, D].

The per-table chunked bmm in bench_onehot.py only reaches ~28% of the measured dense peak
because each GEMM has a reduction dimension of just K=256. Flattening the table axis into
the reduction dimension gives K_eff = T*K = 65,536, which is the shape cuBLAS wants and is
the STRONGEST form of the one-hot argument. The selection matrix is then 24,576 x 65,536
bf16 = 3.22 GB, which fits in 32 GB.

Build and GEMM are timed separately, as the H100 note's complaint was about the build.
"""
import json
import os
import statistics
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8   # noqa: E402
from bench_act3 import T, K, NAP, LO, HI, Q, BLOCK_N, real_cells  # noqa: E402
from bench_onehot import decode, stats  # noqa: E402

N, D = 24576, 1024


@torch.no_grad()
def main():
    pow2_int8.ensure_registered()
    g = torch.Generator(device='cuda').manual_seed(5)
    W8 = torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g)
    cells, _, _, _ = real_cells(N)
    c1, c2, w1, w2 = decode(cells)
    Wd = pow2_int8.stride_tables(W8, D)
    ref = pow2_int8.read_cells(Wd, cells, NAP, D, LO, HI, Q,
                               block_n=BLOCK_N, load16=False).squeeze(1)
    as_ms = stats(lambda: pow2_int8.read_cells(Wd, cells, NAP, D, LO, HI, Q,
                                               block_n=BLOCK_N, load16=False))
    print(f'AS read_cells: {as_ms:.4f} ms')

    Wf = W8.to(torch.bfloat16).contiguous()             # [T*K, D] = the flattened table
    off = (torch.arange(T, device='cuda') * K).view(1, T)
    g1, g2 = c1 + off, c2 + off                          # global cell ids

    S = torch.zeros(N, T * K, device='cuda', dtype=torch.bfloat16)
    rows = torch.arange(N, device='cuda').view(N, 1).expand(N, T)

    def build():
        S.zero_()
        S[rows, g1] = w1.to(torch.bfloat16)
        S[rows, g2] += w2.to(torch.bfloat16)

    torch.cuda.reset_peak_memory_stats()
    build()
    bt = stats(build, iters=6, warmup=2)
    gt = stats(lambda: S @ Wf, iters=6, warmup=2)
    y = (S @ Wf).float()
    pk = torch.cuda.max_memory_allocated() / 1e9
    d = (y - ref).abs()
    sc = ref.abs().max().clamp_min(1)
    flops = 2.0 * N * T * K * D
    print(f'selection matrix: {tuple(S.shape)} bf16 = {S.numel()*2/1e9:.2f} GB')
    print(f'  build  {bt:8.4f} ms')
    print(f'  GEMM   {gt:8.4f} ms   ({flops/(gt*1e-3)/1e12:.1f} TFLOP/s, '
          f'{100*flops/(gt*1e-3)/238.9e12:.0f}% of the 238.9 TFLOP/s measured peak)')
    print(f'  total  {bt+gt:8.4f} ms   vs AS {as_ms:.4f} -> '
          f'{(bt+gt)/as_ms:.1f}x SLOWER  (GEMM alone {gt/as_ms:.1f}x)')
    print(f'  peak memory {pk:.2f} GB')
    print(f'  numerics: max abs {d.max().item():.1f}, max rel {(d.max()/sc).item():.2e}')

    json.dump({'as_ms': as_ms, 'build_ms': bt, 'gemm_ms': gt, 'total_ms': bt + gt,
               'ratio_total': (bt + gt) / as_ms, 'ratio_gemm': gt / as_ms,
               'tflops': flops / (gt * 1e-3), 'peak_gb': pk,
               'max_abs': d.max().item(), 'max_rel': (d.max() / sc).item()},
              open(os.path.join(HERE, 'artifacts', 'onehot_big.json'), 'w'), indent=1)
    print('\nwrote artifacts/onehot_big.json')


if __name__ == '__main__':
    main()
