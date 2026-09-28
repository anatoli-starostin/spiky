"""One-hot GEMM formulation of the LUT read, measured at the study shape on the 5090.

Tests whether the old H100 "14-22x slower" result was an artifact of D=48.

The selection matrix carries the DECODED power-of-two coefficient, not a plain 1: our read
is `acc[j] += (row[j] & m) << sh`, so cell (n, t, c1) contributes `2^sh1 * T[t, c1, j]`.
S_t[n, c] therefore holds 2^sh for the one or two live cells of that (token, table) and 0
elsewhere, and `Y = sum_t S_t @ T_t` reproduces AS exactly in exact arithmetic.

NUMERICS, worth stating because it is not the usual GEMM story: both operands are exactly
representable in bf16 -- the table values are int8 (|v| <= 127, and bf16 holds integers to
256 exactly) and 2^sh is a power of two -- and cuBLAS accumulates bf16 matmuls in fp32,
which is exact for integers below 2^24. The largest |accumulator| we have measured is
1,054,180, well under 2^24. So bit-exactness is NOT obviously lost here, contrary to the
usual "a matmul reassociates" objection. Measured rather than assumed below.

THE int8 TENSOR-CORE PATH CANNOT CARRY THE COEFFICIENT. 2^sh reaches 1024 and does not fit
int8, and the shift is per (token, table, cell) so it cannot be factored out of the GEMM.
A correct int8 formulation would need one 0/1 GEMM per shift plane (8 of them), i.e. 8x the
FLOPs. The int8 number below therefore uses a plain 0/1 selection and is a SPEED-ONLY upper
bound on what int8 tensor cores could do -- it computes the wrong value, and is labelled so.

ncu is unavailable (RmProfilingAdminOnly=1); traffic figures are from launch geometry.
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
from bench_act3 import (T, K, NAP, LO, HI, Q, BLOCK_N, real_cells)  # noqa: E402

N = 24576
DS = [48, 128, 256, 512, 1024]
DISCARD = 15
CHUNK = 16                    # tables per batched-bmm chunk


def stats(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        t.append(s.elapsed_time(e))
    return statistics.median(t)


def decode(cells):
    """(c1, c2, w1, w2) with w = 2^sh, 0 where DISCARD. cells [N, 1, T, 3]."""
    c = cells.squeeze(1)
    c1, c2 = c[..., 0].long(), c[..., 1].long()
    sh = c[..., 2].long()
    sh1, sh2 = sh & 15, sh >> 4
    w1 = torch.where(sh1 == DISCARD, torch.zeros_like(sh1), 1 << sh1).float()
    w2 = torch.where(sh2 == DISCARD, torch.zeros_like(sh2), 1 << sh2).float()
    return c1, c2, w1, w2


def build_sel(c1, c2, w1, w2, t0, nt, dtype):
    """[nt, N, K] selection block for tables [t0, t0+nt). Weighted, so the GEMM is exact."""
    n = c1.shape[0]
    S = torch.zeros(nt, n, K, device=c1.device, dtype=dtype)
    idx = torch.arange(nt, device=c1.device).view(nt, 1)
    rows = torch.arange(n, device=c1.device).view(1, n)
    S[idx, rows, c1[:, t0:t0 + nt].t()] = w1[:, t0:t0 + nt].t().to(dtype)
    S[idx, rows, c2[:, t0:t0 + nt].t()] += w2[:, t0:t0 + nt].t().to(dtype)
    return S


@torch.no_grad()
def run_onehot(Wf, c1, c2, w1, w2, D, dtype, build_only=False, gemm_only=None):
    """sum_t S_t @ T_t, in chunks of CHUNK tables. Returns fp32 [N, D]."""
    n = c1.shape[0]
    Y = torch.zeros(n, D, device='cuda', dtype=torch.float32)
    for t0 in range(0, T, CHUNK):
        nt = min(CHUNK, T - t0)
        S = gemm_only if gemm_only is not None else build_sel(c1, c2, w1, w2, t0, nt, dtype)
        if build_only:
            continue
        # [nt, N, K] @ [nt, K, D] -> [nt, N, D], summed over the chunk's tables
        Y += torch.bmm(S, Wf[t0:t0 + nt]).sum(0).float()
    return Y


@torch.no_grad()
def main():
    pow2_int8.ensure_registered()
    g = torch.Generator(device='cuda').manual_seed(5)
    W8 = torch.randint(-127, 128, (T * K, 1024), device='cuda', dtype=torch.int8, generator=g)
    cells_full, _, _, _ = real_cells(N)
    c1, c2, w1, w2 = decode(cells_full)
    out = {'N': N, 'T': T, 'K': K, 'chunk': CHUNK, 'rows': []}

    print(f'{"D":>6}{"AS ms":>9}{"build ms":>10}{"gemm ms":>9}{"total ms":>10}'
          f'{"vs AS":>8}{"max abs":>11}{"max rel":>10}{"peak GB":>9}  dtype')
    for D in DS:
        Wd = pow2_int8.stride_tables(W8[:, :D].contiguous(), D)
        # read_cells returns [N, H, D]; squeeze to [N, D] so the diff does not broadcast
        # into an [N, N, D] tensor (it tried to allocate 108 GiB the first time).
        ref = pow2_int8.read_cells(Wd, cells_full, NAP, D, LO, HI, Q,
                                   block_n=BLOCK_N, load16=False).squeeze(1)
        as_ms = stats(lambda: pow2_int8.read_cells(Wd, cells_full, NAP, D, LO, HI, Q,
                                                   block_n=BLOCK_N, load16=False))
        for dtype, nm in ((torch.bfloat16, 'bf16'), (torch.float16, 'fp16')):
            Wf = W8[:, :D].view(T, K, D).to(dtype).contiguous()
            torch.cuda.reset_peak_memory_stats()
            bt = stats(lambda: run_onehot(Wf, c1, c2, w1, w2, D, dtype, build_only=True),
                       iters=6)
            tot = stats(lambda: run_onehot(Wf, c1, c2, w1, w2, D, dtype), iters=6)
            y = run_onehot(Wf, c1, c2, w1, w2, D, dtype)
            pk = torch.cuda.max_memory_allocated() / 1e9
            d = (y - ref).abs()
            sc = ref.abs().max().clamp_min(1)
            print(f'{D:>6}{as_ms:>9.4f}{bt:>10.4f}{tot-bt:>9.4f}{tot:>10.4f}'
                  f'{as_ms/tot:>8.4f}{d.max().item():>11.1f}'
                  f'{(d.max()/sc).item():>10.2e}{pk:>9.2f}  {nm}')
            out['rows'].append({'D': D, 'dtype': nm, 'as_ms': as_ms, 'build_ms': bt,
                                'gemm_ms': tot - bt, 'total_ms': tot,
                                'vs_as': as_ms / tot, 'max_abs': d.max().item(),
                                'max_rel': (d.max() / sc).item(), 'peak_gb': pk})
            del Wf, y
            torch.cuda.empty_cache()
        del Wd, ref
        torch.cuda.empty_cache()

    # int8 speed-only upper bound at D=1024
    D = 1024
    print(f'\nint8 SPEED-ONLY upper bound at D={D} (0/1 selection; computes the WRONG value, '
          f'since 2^sh up to 1024 does not fit int8)')
    Wi = W8.view(T, K, D)
    try:
        S = torch.zeros(N, K, device='cuda', dtype=torch.int8)
        rows = torch.arange(N, device='cuda')
        S[rows, c1[:, 0]] = 1

        def f8():
            acc = torch.zeros(N, D, device='cuda', dtype=torch.int32)
            for t in range(T):
                acc += torch._int_mm(S, Wi[t])
            return acc
        ms = stats(f8, iters=4, warmup=2)
        print(f'   int8 GEMM only, {T} tables: {ms:.4f} ms   vs AS 1.4187 -> '
              f'{1.4187/ms:.4f}x')
        out['int8_speed_only_ms'] = ms
    except Exception as ex:
        print(f'   unavailable: {type(ex).__name__}: {str(ex).splitlines()[0][:110]}')
        out['int8_speed_only_ms'] = None

    json.dump(out, open(os.path.join(HERE, 'artifacts', 'onehot_gemm.json'), 'w'), indent=1)
    print('\nwrote artifacts/onehot_gemm.json')


if __name__ == '__main__':
    main()
