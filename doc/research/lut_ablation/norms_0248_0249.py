"""Per-layer FFN-block weight norms, exp_g_0248 vs exp_g_0249: did the TV-shrunk tables get compensated downstream?"""
import torch

RC = '/home/astarostin/projects/spiky/experiments/ffn_replacement/runs_corrected/'
RUNS = (('0248', 'exp_g_0248_B16k_light_learnedmargin_frozeng_tph128_seed1'),
        ('0249', 'exp_g_0249_B16k_light_learnedmargin_frozeng_tv10_tph128_seed1'))
sds = {k: torch.load(RC + d + '/checkpoint.pt', map_location='cpu') for k, d in RUNS}
print('block 0 keys:', [k for k in sds['0248'] if k.startswith('blocks.0.')])
for i in range(6):
    row = []
    for k in sorted(n for n in sds['0248'] if n.startswith(f'blocks.{i}.')
                    and sds['0248'][n].dtype.is_floating_point and sds['0248'][n].numel() > 1):
        a, b = sds['0248'][k].double().norm().item(), sds['0249'][k].double().norm().item()
        row.append(f"{k.split(f'blocks.{i}.')[1]} {a:.4g}->{b:.4g} (x{b / a:.3f})")
    print(f'L{i}: ' + ' | '.join(row))
