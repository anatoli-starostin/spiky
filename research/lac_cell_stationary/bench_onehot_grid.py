"""One-hot GEMM vs AS across N x D, with N = 1, 16, 24576.

Why small N is the interesting test (the amendment's rationale): the selection matrix is
N x K per table, so at N=1 it is 256 values rather than 3.22 GB, and the "selection build
dominates" objection largely evaporates. Meanwhile AS at N=1 is badly under-occupied
(1 block of 1024 threads on a 170-SM GPU) and latency-bound, so its effective bandwidth is
nowhere near the 5,900 GB/s it reaches at N=24,576.

The counter-effect, which is the thing to watch: **the GEMM must read the whole table
whatever N is.** The flattened form [N, T*K] @ [T*K, D] touches all T*K*D weights, i.e.
134 MB in bf16 at D=1024, while AS reads only ~1.33 cells per table -- about 340 KB per
token. That cost is amortised over 24,576 tokens at the top of the ladder and over ONE
token at the bottom. So small N helps the build and hurts the GEMM, and which wins is an
empirical question.

Two variants per cell: `chunk` (per-table bmm, 16 tables at a time) and `flat` (one matmul
with the table axis folded into the reduction dimension). Build and GEMM timed separately.
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
from bench_onehot import decode, stats, build_sel, CHUNK  # noqa: E402

NS = [1, 16, 24576]
DS = [48, 128, 256, 512, 1024]
NMAX = 24576
PEAK_TFLOPS = 238.9


@torch.no_grad()
def main():
    pow2_int8.ensure_registered()
    g = torch.Generator(device='cuda').manual_seed(5)
    W8 = torch.randint(-127, 128, (T * K, 1024), device='cuda', dtype=torch.int8, generator=g)
    cells_all, _, _, _ = real_cells(NMAX)
    out = {'NS': NS, 'DS': DS, 'T': T, 'K': K, 'rows': []}
    off = (torch.arange(T, device='cuda') * K).view(1, T)

    print(f'{"N":>7}{"D":>6}{"AS ms":>9}{"chunk bld":>11}{"chunk gemm":>12}'
          f'{"flat bld":>10}{"flat gemm":>11}{"flat tot":>10}{"flat/AS":>9}'
          f'{"TFLOP/s":>9}{"max rel":>10}')
    for N in NS:
        cells = cells_all[:N].contiguous()
        c1, c2, w1, w2 = decode(cells)
        g1, g2 = c1 + off, c2 + off
        rows = torch.arange(N, device='cuda').view(N, 1).expand(N, T)
        for D in DS:
            Wd = pow2_int8.stride_tables(W8[:, :D].contiguous(), D)
            ref = pow2_int8.read_cells(Wd, cells, NAP, D, LO, HI, Q,
                                       block_n=BLOCK_N, load16=False).squeeze(1)
            as_ms = stats(lambda: pow2_int8.read_cells(Wd, cells, NAP, D, LO, HI, Q,
                                                       block_n=BLOCK_N, load16=False),
                          iters=20)
            # --- chunked per-table bmm ---
            Wc = W8[:, :D].view(T, K, D).to(torch.bfloat16).contiguous()

            def chunk_build():
                for t0 in range(0, T, CHUNK):
                    build_sel(c1, c2, w1, w2, t0, min(CHUNK, T - t0), torch.bfloat16)

            def chunk_all():
                Y = torch.zeros(N, D, device='cuda', dtype=torch.float32)
                for t0 in range(0, T, CHUNK):
                    nt = min(CHUNK, T - t0)
                    S = build_sel(c1, c2, w1, w2, t0, nt, torch.bfloat16)
                    Y += torch.bmm(S, Wc[t0:t0 + nt]).sum(0).float()
                return Y
            cb = stats(chunk_build, iters=6)
            ct = stats(chunk_all, iters=6)
            # --- flattened single matmul ---
            Wf = W8[:, :D].to(torch.bfloat16).contiguous()
            S = torch.zeros(N, T * K, device='cuda', dtype=torch.bfloat16)

            def flat_build():
                S.zero_()
                S[rows, g1] = w1.to(torch.bfloat16)
                S[rows, g2] += w2.to(torch.bfloat16)
            flat_build()
            fb = stats(flat_build, iters=6)
            fg = stats(lambda: S @ Wf, iters=6)
            y = (S @ Wf).float()
            d = (y - ref).abs()
            sc = ref.abs().max().clamp_min(1)
            flops = 2.0 * N * T * K * D
            tf = flops / (fg * 1e-3) / 1e12
            print(f'{N:>7}{D:>6}{as_ms:>9.4f}{cb:>11.4f}{ct-cb:>12.4f}'
                  f'{fb:>10.4f}{fg:>11.4f}{fb+fg:>10.4f}{as_ms/(fb+fg):>9.4f}'
                  f'{tf:>9.1f}{(d.max()/sc).item():>10.2e}')
            out['rows'].append({'N': N, 'D': D, 'as_ms': as_ms,
                                'chunk_build_ms': cb, 'chunk_gemm_ms': ct - cb,
                                'chunk_total_ms': ct, 'chunk_vs_as': as_ms / ct,
                                'flat_build_ms': fb, 'flat_gemm_ms': fg,
                                'flat_total_ms': fb + fg, 'flat_vs_as': as_ms / (fb + fg),
                                'flat_tflops': tf, 'max_rel': (d.max() / sc).item(),
                                'max_abs': d.max().item(),
                                'table_bytes_bf16': T * K * D * 2,
                                'as_table_bytes': 1.3272 * T * D * N})
            del Wd, ref, Wc, Wf, S, y
            torch.cuda.empty_cache()
        del cells, c1, c2, w1, w2, g1, g2, rows
        torch.cuda.empty_cache()

    print('\nWINS FOR ONE-HOT (flat/AS > 1 means the GEMM is FASTER):')
    wins = [r for r in out['rows'] if r['flat_vs_as'] > 1.0]
    if wins:
        for r in wins:
            print(f'   N={r["N"]:<6} D={r["D"]:<5} GEMM {r["flat_total_ms"]:.4f} ms vs '
                  f'AS {r["as_ms"]:.4f} ms -> {r["flat_vs_as"]:.3f}x FASTER')
    else:
        print('   none -- AS wins every cell of the grid')
    json.dump(out, open(os.path.join(HERE, 'artifacts', 'onehot_grid.json'), 'w'), indent=1)
    print('\nwrote artifacts/onehot_grid.json')


if __name__ == '__main__':
    main()
