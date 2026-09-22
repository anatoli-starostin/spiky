"""Print every parameter of the L=4 config and which weight-decay group it lands in.

The partition is decided by IDENTITY, not by name matching: the table tensor is fetched as
inner_lut(block).tables and its id() is compared, so a rename or a same-named tensor elsewhere cannot
put the wrong thing in the decayed group.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from autoencoder import Autoencoder, table_param_ids  # noqa: E402

CFG = dict(width=128, depth_L=2, kind='lut', n_tables=128, device='cuda', seed=0, residual=False,
           block_norm='layernorm', lut_impl='compression', norm_position='pre', n_blocks=4,
           inner_out=-1, inner_in=128, final_norm=True)


def main():
    m = Autoencoder(784, **CFG)
    ids = table_param_ids(m)
    a, b = [], []
    for n, p in m.named_parameters():
        (a if id(p) in ids else b).append((n, tuple(p.shape), p.numel()))
    print(f'GROUP A -- decayed, the LUT table values ({len(a)} tensors)')
    for n, s, c in a:
        print(f'  {n:<28}{str(s):<22}{c:>12,}')
    print(f'  total {sum(c for _, _, c in a):,}')
    print(f'\nGROUP B -- wd = 0, everything else ({len(b)} tensors)')
    for n, s, c in b:
        print(f'  {n:<28}{str(s):<22}{c:>12,}')
    print(f'  total {sum(c for _, _, c in b):,}')
    tot = sum(p.numel() for p in m.parameters())
    print(f'\nall parameters {tot:,}; A is {100*sum(c for _, _, c in a)/tot:.2f}% of them')
    assert len(a) + len(b) == len(list(m.parameters())), 'partition is not a partition'
    assert sum(c for _, _, c in a) + sum(c for _, _, c in b) == tot
    print('partition covers every parameter exactly once: OK')


if __name__ == '__main__':
    main()
