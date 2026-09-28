"""Verbose rebuild (captures -Xptxas -v), SASS RED/ATOM check, and the refusal demo."""
import os
import re
import subprocess
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scatter  # noqa: E402

T, K, D = 256, 256, 1024


def main():
    scatter.mod(verbose=True, fresh=True)
    print('BUILD DONE')

    so = os.path.join(scatter.BUILD, 'lac_scatter.so')
    sass = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '-sass', so],
                          capture_output=True, text=True).stdout
    print('\nSASS ATOMIC FORM (arm A must be RED, fire-and-forget, not ATOM)')
    print(f'{"kernel":<30}{"RED":>6}{"ATOM":>6}{"ATOMS":>7}{"verdict":>28}')
    for part in re.split(r'Function : ', sass)[1:]:
        nm = part.split('\n', 1)[0]
        nums = [int(x) for x in re.findall(r'Li(\d+)E', nm)]
        if 'scatter_global' in nm:
            key = f'armA L={nums[0]}'
        elif 'scatter_shared' in nm:
            key = f'armB Bt={nums[0]} L={nums[1]} pad={nums[2]}'
        else:
            continue
        # sm_120 mnemonics: REDG = global reduction with NO return value (what arm A must emit),
        # ATOMG = global atomic that returns the old value, ATOMS = shared-memory atomic.
        red = len(re.findall(r'\bREDG[.A-Z0-9]*', part))
        atom = len(re.findall(r'\bATOMG[.A-Z0-9]*', part))
        atoms = len(re.findall(r'\bATOMS[.A-Z0-9]*', part))
        if key.startswith('armA'):
            v = 'RED (fire-and-forget)' if red and not atom else (
                'ATOM -- returns a value!' if atom else 'no atomic found')
        else:
            v = 'ATOMS = shared atomic' if atoms else ('unexpected: ' + str((red, atom)))
        print(f'{key:<30}{red:>6}{atom:>6}{atoms:>7}{v:>28}')

    print('\nREFUSAL DEMO (no silent fallback)')
    g = torch.Generator(device='cuda').manual_seed(1)
    W = torch.randint(-127, 128, (T * K, D), device='cuda', dtype=torch.int8, generator=g)
    cells = torch.zeros(128, 1, T, 3, device='cuda', dtype=torch.uint8)
    for desc, fn in [
        ('arm A, L=5 (does not divide D=1024)', lambda: scatter.run_a(W, cells, D, 5)),
        ('arm A, L=3 (no instantiation)', lambda: scatter.run_a(W, cells, D, 3)),
        ('arm A, threads=1056 (>1024)', lambda: scatter.run_a(W, cells, D, 4, threads=1056)),
        ('arm B, Bt=32 (smem 131,072 B, over budget)',
         lambda: scatter.run_b(W, cells, D, 4, 32)),
        ('arm B, Bt=16 L=32 (no instantiation)', lambda: scatter.run_b(W, cells, D, 32, 16)),
    ]:
        try:
            fn()
            print(f'  {desc:<46} NO ERROR  <-- unexpected')
        except Exception as ex:
            print(f'  {desc:<46} {str(ex).splitlines()[0][:110]}')


if __name__ == '__main__':
    main()
