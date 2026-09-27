"""Quick single-iteration timing probe, to size the real benchmark's time budget."""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lac  # noqa: E402
from bench import make_shape_B  # noqa: E402


def one(fn):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


def main():
    lac.mod()
    for B in (256, 1024, 24576):
        T, J, C = make_shape_B(B)
        y = torch.zeros(B, T.shape[2], device='cuda')
        print(f'B={B}')
        print(f'   gather tsplit=1      {one(lambda: lac.run_gather(T, J, C, y=y)):9.3f} ms')
        for K, M, TB, CLU in ((32, 1, 4, 1), (16, 2, 4, 1), (32, 8, 4, 1), (32, 16, 4, 1),
                              (32, 32, 4, 1), (16, 32, 4, 1), (64, 32, 2, 1),
                              (32, 32, 1, 1), (32, 16, 4, 8)):
            t = one(lambda: lac.run_v1(T, J, C, K=K, M=M, TB=TB, CLU=CLU, y=y))
            print(f'   cs_v1 K={K} M={M} TB={TB} CLU={CLU}   {t:9.3f} ms')
        for K, M in ((4, 48), (4, 128), (8, 96), (16, 96)):
            t = one(lambda: lac.run_v0(T, J, C, K=K, M=M, y=y))
            print(f'   cs_v0 K={K} M={M}        {t:9.3f} ms')
        del T, J, C, y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
