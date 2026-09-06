"""Do the margins widen during training — would bounded's attenuation have healed itself?"""
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.fast_multi_head_lut import _confidence_score

init = torch.load('/tmp/margins_anchor.pt')
trained = torch.load('/tmp/margins_anchor_trained.pt')

print(f"{'':<26}{'|d| median':>12}{'|d| mean':>11}"
      f"{'bounded':>11}{'bnorm':>10}{'margin':>10}")
for name, blob in (('at init', init), ('after 4,000 steps', trained)):
    d = blob['d']
    b = _confidence_score(d, 'bounded').mean().item()
    n = _confidence_score(d, 'bounded_norm').mean().item()
    g = _confidence_score(d, 'margin').mean().item()
    print(f'   {name:<23}{d.abs().median():>12.4f}{d.abs().mean():>11.4f}'
          f'{b:>11.4f}{n:>10.4f}{g:>10.4f}')

di, dt = init['d'].abs(), trained['d'].abs()
bi = _confidence_score(init['d'], 'bounded').mean().item()
bt = _confidence_score(trained['d'], 'bounded').mean().item()
print(f'\n   margins grew {dt.median() / di.median():.2f}x (median) over 4,000 steps')
print(f'   bounded score rose {bt / bi:.2f}x, from {bi:.4f} to {bt:.4f}')
print(f'   -> attenuation eased from {1/bi:.1f}x to {1/bt:.1f}x, still {1/bt:.1f}x at the END')
print('\n   For bounded to reach ~0.5 the median margin would need to be '
      f'{torch.log(torch.tensor(0.5)).item() / 8:.4f} in logsigmoid terms, i.e.')
target = -torch.log(torch.exp(torch.log(torch.tensor(0.5)) / 8).reciprocal() - 1) / 2
print(f'   |d| ~ {target.item():.3f}, against the {dt.median():.3f} actually reached — '
      f'a further {target.item() / dt.median():.2f}x.')
print('\n   And this is the GATE-OFF run\'s trajectory. Under the gate the FFN contributes')
print('   ~18x less, so the compressed code has far less pressure to spread at all.')
