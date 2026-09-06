"""BH4 throughput vs the Light runs, and numerical proof that the fast path is the same.

The reference implementation is written for clarity: fwht is a Python loop over log2(d)
butterfly stages, each allocating a [N, H, d] temporary. At our shapes that made the layer
memory-bound. This measures the cost, and asserts the replacement is arithmetically the
reference -- forward AND gradient -- rather than assuming it.

    python bench_bh4.py
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
sys.path.insert(0, '/tmp/claude-1000')

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def bench(fn, n=12, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if DEV == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize() if DEV == 'cuda' else None
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    from spiky.lutorch.bh4_multi_head_lut import (BH4MultiHead, fwht, hadamard_matrix)

    print('=' * 88)
    print('NUMERICAL EQUIVALENCE -- the fast Hadamard must BE the reference one')
    print('=' * 88)
    for n in (256, 1024):
        H = hadamard_matrix(n)
        x = torch.randn(64, 3, n, dtype=torch.float64)
        Hd = H.double()
        a, b = fwht(x), x @ Hd
        print(f'   n={n:>5}  max|x@H - fwht(x)| = {(a - b).abs().max():.3e}   '
              f'H involutory: max|HH-I| = {(Hd @ Hd - torch.eye(n, dtype=torch.float64)).abs().max():.3e}')

    # gradient equivalence through the whole multi-head stack, against the reference module
    try:
        from lookup_ffn import BH4 as RefBH4
    except ImportError:
        RefBH4 = None
        print('   *** reference module not importable; skipping the module comparison')
    if RefBH4 is not None:
        Hh, D, BLK, NF = 4, 256, 4, 4
        ours = BH4MultiHead(D, Hh, block=BLK, n_factors=NF, random_seed=0).double()
        refs = [RefBH4(D, block=BLK, n_factors=NF).double() for _ in range(Hh)]
        with torch.no_grad():
            for h, r in enumerate(refs):
                r.blocks.copy_(ours.blocks[h])
        x1 = torch.randn(33, Hh, D, dtype=torch.float64, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        y1 = ours(x1)
        y2 = torch.stack([refs[h](x2[:, h]) for h in range(Hh)], dim=1)
        print(f'\n   forward  max|ours - reference| = {(y1 - y2).abs().max():.3e}')
        (y1 * y1).sum().backward()
        (y2 * y2).sum().backward()
        gb_ref = torch.stack([r.blocks.grad for r in refs])
        print(f'   grad x   max|diff| = {(x1.grad - x2.grad).abs().max():.3e}')
        print(f'   grad B   max|diff| = {(ours.blocks.grad - gb_ref).abs().max():.3e}')

    if DEV != 'cuda':
        print('\n(no GPU: skipping throughput)')
        return

    print('\n' + '=' * 88)
    print('THROUGHPUT -- one training step (fwd+bwd) at the real shapes')
    print('=' * 88)
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()
    runs = [
        ('exp_g_0190 Light (nap8)', 'exp_g_0190_B16k_light_bnorm_tph128_znorm_seed1'),
        ('exp_g_0192 BH4  (nap7)', 'exp_g_0192_B16k_bh4_margin_tph128_nap7_seed1'),
    ]
    idx = torch.randint(0, vocab, (12, 512), device=DEV)
    for label, run in runs:
        cfg = json.load(open(os.path.join(HERE, run, 'config.json')))
        torch.manual_seed(cfg['random_seed'])
        m = build_model(cfg, vocab, device=DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def step():
            opt.zero_grad(set_to_none=True)
            loss = m(idx, idx)
            loss.backward()
            opt.step()

        ms = bench(step)
        # grad_accum is 4 in these configs, so a logged "step" is 4 of these
        print(f'   {label:<26} {ms:>8.1f} ms/micro-batch   '
              f'{ms * 4 / 1000:>6.3f} s/step   '
              f'16k steps = {ms * 4 * 16000 / 3600000:>5.2f} h (+ evals)')
        del m, opt
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
