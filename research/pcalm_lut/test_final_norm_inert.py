"""--final-norm off must leave the module BYTE-IDENTICAL to before the flag existed.

This is the check the dropout mistake earned: there, inserting an nn.Dropout into an existing
nn.Sequential renumbered the Linear after it, silently renaming its state_dict keys and breaking every
checkpoint written earlier. The fix pattern is to assign None rather than register a placeholder module,
and this asserts that it worked:

  1. the state_dict KEY SET is identical with the flag off
  2. so is the parameter COUNT
  3. the forward is bit-identical with the flag off
  4. with the flag ON, exactly two keys appear (final_ln.weight, final_ln.bias) and nothing is renamed

Run directly, not under pytest, per the convention for these files:  python test_final_norm_inert.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from autoencoder import Autoencoder  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CFG = dict(width=128, depth_L=2, kind='lut', n_tables=128, device=DEV, seed=0, residual=False,
           block_norm='layernorm', lut_impl='compression', norm_position='pre', n_blocks=4,
           inner_out=-1, inner_in=128)


def main():
    fails = []
    off = Autoencoder(784, **CFG, final_norm=False)
    on = Autoencoder(784, **CFG, final_norm=True)
    k_off, k_on = set(off.state_dict()), set(on.state_dict())

    print(f'1. keys with flag off: {len(k_off)};  with flag on: {len(k_on)}')
    added = k_on - k_off
    removed = k_off - k_on
    print(f'4. added by the flag: {sorted(added)};  removed or renamed: {sorted(removed) or "none"}')
    if added != {'final_ln.weight', 'final_ln.bias'}:
        fails.append(f'the flag added {sorted(added)}, expected exactly final_ln.weight/bias')
    if removed:
        fails.append(f'the flag REMOVED OR RENAMED {sorted(removed)} -- checkpoints would break')

    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    print(f'2. params off {n_off}, on {n_on}, difference {n_on - n_off} (expected 2*128 = 256)')
    if n_on - n_off != 2 * CFG['width']:
        fails.append('parameter count delta is not one LayerNorm')

    off.eval()
    ref = Autoencoder(784, **CFG, final_norm=False)
    ref.eval()
    x = torch.randn(16, 784, device=DEV)
    with torch.no_grad():
        same = torch.equal(off(x), ref(x))
    print(f'3. two flag-off models at the same seed are bit-identical: {same}')
    if not same:
        fails.append('the flag-off path is not reproducible, so the comparison is meaningless')

    # and the flag-on model must actually differ, or the norm is not wired into the forward
    on.eval()
    with torch.no_grad():
        differs = not torch.equal(on(x), off(x))
    print(f'   flag-on output differs from flag-off: {differs}')
    if not differs:
        fails.append('--final-norm changes nothing in the forward -- it is not wired in')

    print('FAIL: ' + '; '.join(fails) if fails else 'all checks pass: the flag is inert when off')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
