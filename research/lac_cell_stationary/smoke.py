"""Build the extension and check every kernel variant against the torch reference
on a small synthetic problem. Fast; run this before anything else."""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lac  # noqa: E402

V0 = [(1, 192), (1, 128), (1, 64), (2, 96), (2, 64), (4, 48), (4, 64), (4, 32)]
V1 = [(8, 4, 4, 1), (8, 2, 4, 1), (8, 1, 4, 1),
      (16, 4, 4, 1), (16, 2, 4, 1), (16, 1, 4, 1),
      (32, 2, 4, 1), (32, 1, 4, 1),
      (32, 1, 2, 1), (64, 1, 2, 1), (64, 1, 1, 1), (32, 2, 1, 1),
      (16, 4, 2, 1), (8, 4, 2, 1),
      (32, 1, 4, 2), (32, 1, 4, 4), (32, 1, 4, 8),
      (16, 2, 4, 4), (16, 2, 4, 8), (16, 4, 4, 8)]


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    NT, R, N, B = 64, 256, 128, 37          # small, and B deliberately not a multiple of M
    T = torch.randint(-127, 128, (NT, R, N), device=dev, dtype=torch.int8)
    J = torch.randint(0, R, (B, NT), device=dev, dtype=torch.uint8)
    C = torch.randn(B, NT, device=dev, dtype=torch.float32)

    print('building (this compiles ~30 kernel instantiations, takes a minute) ...')
    lac.mod(verbose=True)
    print('build ok\n')

    for use_coef in (True, False):
        ref = lac.reference(T, J, C, use_coef=use_coef)
        scale = ref.abs().max().item()
        tag = 'c=real' if use_coef else 'c=1'
        bad = 0
        for ts in (1, 2, 4, 8):
            y = lac.run_gather(T, J, C, tsplit=ts, use_coef=use_coef)
            e = (y - ref).abs().max().item()
            ok = e / scale < 1e-5
            bad += not ok
            print(f'  {tag}  gather   tsplit={ts:<3} max_abs={e:.3e}  rel={e/scale:.2e}  '
                  f'{"OK" if ok else "FAIL"}')
        for K, M in V0:
            y = lac.run_v0(T, J, C, K=K, M=M, use_coef=use_coef)
            e = (y - ref).abs().max().item()
            ok = e / scale < 1e-5
            bad += not ok
            print(f'  {tag}  cs_v0    K={K:<3} M={M:<4} max_abs={e:.3e}  rel={e/scale:.2e}  '
                  f'{"OK" if ok else "FAIL"}')
        for K, M, TB, CLU in V1:
            if N % K or NT % (TB * CLU):
                continue
            y = lac.run_v1(T, J, C, K=K, M=M, TB=TB, CLU=CLU, use_coef=use_coef)
            e = (y - ref).abs().max().item()
            ok = e / scale < 1e-5
            bad += not ok
            print(f'  {tag}  cs_v1    K={K:<3} M={M} TB={TB} CLU={CLU}  max_abs={e:.3e}  '
                  f'rel={e/scale:.2e}  {"OK" if ok else "FAIL"}')
        print()
    print('FAILURES:', bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
