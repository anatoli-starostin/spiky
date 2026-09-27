"""Print the real shape of the LUT layers in a trained checkpoint.

Step 0 of the task: verify the config against the repo instead of assuming it.
"""
import json
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
sys.path.insert(0, os.path.join(SPIKY, 'experiments/ffn_replacement/tools'))

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402
from model_build import build_model                                # noqa: E402

RUN = os.path.join(SPIKY, 'experiments/ffn_replacement/runs_corrected',
                   'exp_n_0196_light_margin_znorm_nap8_tph256_seed1')


def main():
    cfg = json.load(open(os.path.join(RUN, 'config.json')))
    model = build_model(cfg, cfg['tokenizer_vocab_size'], device='cuda')
    sd = torch.load(os.path.join(RUN, 'checkpoint.pt'), map_location='cuda')
    sd = sd.get('model', sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'loaded: {len(missing)} missing, {len(unexpected)} unexpected')
    if missing:
        print('  missing[:5]:', list(missing)[:5])
    if unexpected:
        print('  unexpected[:5]:', list(unexpected)[:5])

    luts = [(n, m) for n, m in model.named_modules() if isinstance(m, LightMultiHeadLUT)]
    print(f'\n{len(luts)} LightMultiHeadLUT modules')
    n, m = luts[0]
    print(f'\nfirst one: {n}')
    for attr in ('input_dim', 'output_dim', 'n_tables', 'table_size', 'n_anchor_pairs',
                 'n_heads', 'tables_per_head', 'multi_head_input', 'read_top_n',
                 'confidence_form', 'confidence_gain', 'forward_mode'):
        print(f'   {attr:<20} {getattr(m, attr, "-")}')
    for b in ('tables', 'anchor_a', 'anchor_b', 'powers', 'table_offset'):
        t = getattr(m, b, None)
        print(f'   {b:<20} {tuple(t.shape) if t is not None else "-"}'
              f'{"  dtype " + str(t.dtype) if t is not None else ""}')
    tb = m.tables
    print(f'\n   tables: {tb.numel():,} values, '
          f'{tb.numel()/2**20:.2f} MiB as int8, {tb.numel()*4/2**20:.2f} MiB as fp32')
    print(f'   abs max {tb.abs().max().item():.5f}   std {tb.std().item():.5f}')
    print('\nparent module chain:')
    parent = dict(model.named_modules())[n.rsplit('.', 1)[0]]
    print('  ', type(parent).__name__)
    for cn, cm in parent.named_children():
        extra = ''
        if hasattr(cm, 'weight') and cm.weight is not None:
            extra = f' weight {tuple(cm.weight.shape)}'
        print(f'     {cn:<16} {type(cm).__name__}{extra}')


if __name__ == '__main__':
    main()
