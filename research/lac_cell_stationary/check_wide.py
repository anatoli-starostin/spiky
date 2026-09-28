"""Does the widened dispatch actually build and produce the right answer at D=1024?

Correctness and dispatch only -- no timing, no sweep. Checks that block_n=16 with
UPR = 1024/16 = 64 (1024 threads per block) compiles, launches, and matches a torch
reference of the same shift-add, and that the refusal paths still refuse.
"""
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
from spiky.lutorch import pow2_int8, pow2_read  # noqa: E402

H, T, NAP, K = 1, 256, 8, 256
N = 64


def reference(packed, cells, D):
    """The kernel's accumulation, in torch: acc[j] += (row[j] & m) << s over 2T cells."""
    NN = cells.shape[0]
    acc = torch.zeros(NN, H, D, dtype=torch.int32, device=packed.device)
    off = (torch.arange(T, device=packed.device) * K).view(1, 1, T)
    for r in range(2):
        idx = (cells[..., r].to(torch.int64) + off).reshape(-1)        # [NN*H*T]
        sh = ((cells[..., 2] >> (4 * r)) & 15).to(torch.int32)         # [NN,H,T]
        rows = packed[idx].to(torch.int32).view(NN, H, T, D)
        keep = (sh != pow2_int8.DISCARD).view(NN, H, T, 1)
        acc += torch.where(keep, rows << sh.view(NN, H, T, 1), torch.zeros_like(rows)).sum(2)
    return acc.float()


@torch.no_grad()
def main():
    print(f'extension: {pow2_int8.available()[1]}')
    torch.manual_seed(3)
    for D in (1024, 512, 256):
        upr = D // 16
        packed = torch.randint(-127, 128, (H * T * K, D), device='cuda', dtype=torch.int8)
        tab = pow2_int8.stride_tables(packed, D)
        c1 = torch.randint(0, K, (N, H, T), device='cuda', dtype=torch.int64)
        c2 = torch.randint(0, K, (N, H, T), device='cuda', dtype=torch.int64)
        sh1 = torch.randint(3, 11, (N, H, T), device='cuda', dtype=torch.int64)
        sh2 = torch.randint(3, 11, (N, H, T), device='cuda', dtype=torch.int64)
        # force a realistic share of discards through both codes
        sh1 = torch.where(torch.rand_like(sh1.float()) < 0.5, torch.full_like(sh1, 15), sh1)
        sh2 = torch.where((sh1 == 15) | (torch.rand_like(sh2.float()) < 0.3),
                          torch.full_like(sh2, 15), sh2)
        cells = torch.stack([c1, c2, sh1 | (sh2 << 4)], -1).to(torch.uint8).contiguous()

        ref = reference(tab, cells, D)
        for load16 in (True, False):
            got = pow2_int8.read_cells(tab, cells, NAP, D, -3, 4, 3, block_n=16, load16=load16)
            e = (got - ref).abs().max().item()
            print(f'  D={D:<5} upr={upr:<3} block_n=16 -> {16*upr:>4} threads  '
                  f'load16={str(load16):<5} max|diff| {e:.1f}  {"OK" if e == 0 else "FAIL"}')
        del packed, tab, cells, ref
        torch.cuda.empty_cache()

    print('\nrefusal paths still refuse:')
    packed = torch.zeros(H * T * K, 1024, device='cuda', dtype=torch.int8)
    cells = torch.zeros(4, H, T, 3, device='cuda', dtype=torch.uint8)
    for bn in (32, 64, 128):
        try:
            pow2_int8.read_cells(packed, cells, NAP, 1024, -3, 4, 3, block_n=bn)
            print(f'  block_n={bn} D=1024  NO ERROR  <-- unexpected')
        except Exception as ex:
            print(f'  block_n={bn} D=1024  refused: {str(ex).splitlines()[0][:96]}')
    try:
        p2 = torch.zeros(H * T * K, 1040, device='cuda', dtype=torch.int8)
        c = torch.zeros(4, H, T, 3, device='cuda', dtype=torch.uint8)
        pow2_int8.read_cells(p2, c, NAP, 1040, -3, 4, 3, block_n=16)
        print('  block_n=16 D=1040 (upr 65)  NO ERROR  <-- unexpected')
    except Exception as ex:
        print(f'  block_n=16 D=1040 (upr 65)  refused: {str(ex).splitlines()[0][:96]}')


if __name__ == '__main__':
    main()
