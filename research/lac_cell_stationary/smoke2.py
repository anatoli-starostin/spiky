"""Correctness + first timing for cs_v2 (conflict-free reduction) and cs_v3 (G tables
per thread), in both table layouts."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lac  # noqa: E402

V2 = [(16, 32, 4), (32, 32, 4), (64, 32, 4), (32, 32, 2), (64, 32, 2), (128, 32, 2),
      (64, 16, 4), (128, 16, 4), (128, 32, 1), (64, 32, 1), (32, 16, 4), (16, 32, 2)]
V3 = [(32, 32, 1), (32, 32, 4), (32, 32, 8), (64, 32, 8), (16, 32, 8), (16, 32, 32),
      (8, 32, 32), (8, 32, 64), (4, 32, 128), (4, 32, 64), (8, 32, 16), (16, 16, 32)]


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    NT, R, N, B = 128, 256, 128, 71
    T = torch.randint(-127, 128, (NT, R, N), device=dev, dtype=torch.int8)
    Tt = lac.transpose_tables(T)
    J = torch.randint(0, R, (B, NT), device=dev, dtype=torch.uint8)
    C = torch.randn(B, NT, device=dev)

    lac.mod(verbose=True)
    bad = 0
    for use_coef in (True, False):
        ref = lac.reference(T, J, C, use_coef=use_coef)
        sc = ref.abs().max().item()
        tag = 'c=real' if use_coef else 'c=1   '
        for K, M, TB in V2:
            if N % K or NT % TB:
                continue
            for tr in (0, 1):
                y = lac.run_v2(Tt if tr else T, J, C, K=K, M=M, TB=TB, trans=tr,
                               use_coef=use_coef)
                e = (y - ref).abs().max().item()
                ok = e / sc < 1e-5
                bad += not ok
                print(f'  {tag} cs_v2 K={K:<4}M={M:<3}TB={TB} trans={tr}  '
                      f'rel={e/sc:.2e} {"OK" if ok else "FAIL"}')
        for K, M, G in V3:
            if N % K or NT % G:
                continue
            for tr in (0, 1):
                y = lac.run_v3(Tt if tr else T, J, C, K=K, M=M, G=G, trans=tr,
                               use_coef=use_coef)
                e = (y - ref).abs().max().item()
                ok = e / sc < 1e-5
                bad += not ok
                print(f'  {tag} cs_v3 K={K:<4}M={M:<3}G={G:<4}trans={tr}  '
                      f'rel={e/sc:.2e} {"OK" if ok else "FAIL"}')
    print('FAILURES:', bad)


if __name__ == '__main__':
    main()
