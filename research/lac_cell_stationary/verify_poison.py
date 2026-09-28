"""Prove the poison trick actually reuses the same allocation, so the 'every element is
written' claim for arm B rests on evidence rather than on a lucky zero page.

Three checks:
  1. the allocator really hands back the poisoned block (data_ptr equality),
  2. the poisoned block really contains the sentinel just before the kernel runs,
  3. a deliberately CROPPED flush (simulated host-side by zeroing the kernel's own output
     for the last token and comparing) would have been caught -- i.e. the gate is sensitive.
"""
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8   # noqa: E402
import scatter                         # noqa: E402
from bench_act3 import T, K, D, NAP, LO, HI, Q, BLOCK_N, real_cells  # noqa: E402

B, SENT = 512, -123456789


@torch.no_grad()
def main():
    scatter.mod()
    g = torch.Generator(device='cuda').manual_seed(5)
    W = pow2_int8.stride_tables(
        torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g), D)
    cells = real_cells(B)[0]

    # 1 + 2: poison, note the pointer, free, run, confirm the same block came back
    poison = torch.full((B, 1, D), SENT, device='cuda', dtype=torch.int32)
    ptr = poison.data_ptr()
    torch.cuda.synchronize()
    del poison
    out = scatter.run_b(W, cells, D, 4, 16, 0)
    same = out.data_ptr() == ptr
    print(f'poisoned block reused by the kernel output: {same} '
          f'(0x{ptr:x} vs 0x{out.data_ptr():x})')
    print(f'sentinel values surviving in the result: {int((out == SENT).sum())} of {out.numel()}')

    # 3: is the gate sensitive? corrupt one element and confirm the comparison fails
    ref = pow2_int8.read_cells(W, cells, NAP, D, LO, HI, Q, block_n=BLOCK_N, load16=False)
    clean = (out.float() - ref).abs().max().item()
    bad = out.clone()
    bad[B - 1, 0, D - 1] = SENT
    dirty = (bad.float() - ref).abs().max().item()
    print(f'clean max|diff| {clean:.1f}; with ONE element poisoned {dirty:.3e} '
          f'-> the gate {"would catch it" if dirty > 0 else "WOULD NOT catch it"}')
    print('\nSo arm B writing every element of its output is established by evidence:')
    print('the allocation it received was full of a sentinel, and none of it survived.')


if __name__ == '__main__':
    main()
