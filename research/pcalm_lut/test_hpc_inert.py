"""--deep-supervision hpc must add NO parameters and rename nothing.

Same check the --final-norm flag got, for the same reason: a previous flag inserted a module into an
existing container, renumbered the layer after it and silently broke every earlier checkpoint. HPC
decodes every level through the SHARED final_ln + dec, so by construction it adds no module -- this
asserts that construction actually holds.

  1. identical state_dict KEY SET, none added, none renamed
  2. identical parameter COUNT
  3. bit-identical model at the same seed, and a bit-identical plain forward
  4. levels() really returns n_blocks predictions and their SUM is what forward() returns
  5. the level-0 prediction is NOT the full reconstruction (i.e. deeper levels contribute)

Run directly:  python test_hpc_inert.py
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
           inner_out=-1, inner_in=128, final_norm=True)


def main():
    fails = []
    off = Autoencoder(784, **CFG, deep_supervision='none')
    on = Autoencoder(784, **CFG, deep_supervision='hpc')
    k_off, k_on = set(off.state_dict()), set(on.state_dict())
    print(f'1. keys off {len(k_off)}, on {len(k_on)}; added {sorted(k_on - k_off) or "none"}; '
          f'removed/renamed {sorted(k_off - k_on) or "none"}')
    if k_off != k_on:
        fails.append('the flag changed the state_dict key set')

    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    print(f'2. params off {n_off}, on {n_on}, difference {n_on - n_off} (expected 0)')
    if n_off != n_on:
        fails.append('the flag changed the parameter count')

    off.eval()
    on.eval()
    x = torch.randn(16, 784, device=DEV)
    with torch.no_grad():
        same_w = all(torch.equal(a, b) for a, b in zip(off.state_dict().values(),
                                                       on.state_dict().values()))
        # the plain forward path of the hpc model, reached by flipping the attribute, must match
        on.deep_supervision = 'none'
        same_f = torch.equal(off(x), on(x))
        on.deep_supervision = 'hpc'
    print(f'3. identical weights at the same seed: {same_w}; identical plain forward: {same_f}')
    if not (same_w and same_f):
        fails.append('the flag-off path is not bit-identical')

    with torch.no_grad():
        preds = on.levels(x)
        summed, fwd = sum(preds), on(x)
    print(f'4. levels() returns {len(preds)} predictions (expected {CFG["n_blocks"]}); '
          f'sum == forward(): {torch.equal(summed, fwd)}')
    if len(preds) != CFG['n_blocks'] or not torch.equal(summed, fwd):
        fails.append('forward() is not the sum of the level predictions')

    diff = float((preds[0] - fwd).abs().mean())
    print(f'5. mean |level 0 - full reconstruction| = {diff:.4f} (must be > 0, or deeper levels '
          f'contribute nothing)')
    if diff == 0:
        fails.append('level 0 already equals the full reconstruction')

    print('FAIL: ' + '; '.join(fails) if fails else 'all checks pass: hpc adds no parameters')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
