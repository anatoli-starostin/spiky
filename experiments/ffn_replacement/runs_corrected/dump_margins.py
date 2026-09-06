"""Dump the real routing margins |d| at the anchor sizing to a file, once.

Every confidence-form question downstream (which normalisation, how selective, how it
behaves as nap varies) is a question about the SAME margins, so capture them once on CPU
and iterate against the file instead of rebuilding the model each time.

    python dump_margins.py            ->  /tmp/margins_anchor.pt
"""
import json
import os
import sys

import torch

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RC = os.path.join(FR, 'runs_corrected')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))

from nanochat.common import get_base_dir                                   # noqa: E402
from nanochat.tokenizer import RustBPETokenizer                            # noqa: E402
from nanochat.dataloader import (                                          # noqa: E402
    tokenizing_distributed_data_loader_bos_bestfit)
from model_build import build_model                                        # noqa: E402

DEV = 'cpu'          # the GPU belongs to whatever is training
TRAINED = '--trained' in sys.argv
OUT = '/tmp/margins_anchor_trained.pt' if TRAINED else '/tmp/margins_anchor.pt'
RUN = 'sweep_s05_dout48_H4_tph256_c256_din32'


def main():
    cfg = json.load(open(os.path.join(RC, RUN, 'config.json')))
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    ld = tokenizing_distributed_data_loader_bos_bestfit(tok, 2, 512, split='val', device=DEV)
    x_ids, _ = next(iter(ld))

    torch.manual_seed(cfg['random_seed'])
    m = build_model(cfg, tok.get_vocab_size(), device=DEV)
    if TRAINED:
        # The SAME model after 4,000 steps: answers whether the margins widen during
        # training, i.e. whether bounded's attenuation would have healed itself.
        ck = os.path.join(RC, RUN, 'checkpoint.pt')
        miss, unexp = m.load_state_dict(torch.load(ck, map_location=DEV), strict=False)
        print(f'loaded {ck}  (missing {len(miss)}, unexpected {len(unexp)})')
    m.eval()

    per_block = []
    for blk in m.blocks:
        ffn = blk.ffn
        rec = {}

        def hook(mod, inp, _out, rec=rec, ffn=ffn):
            z = ffn.compress(inp[0]).view(inp[0].shape[0], ffn.n_heads, ffn.inner_in_dim)
            lut = ffn.lut_batched
            a, b = lut.soft_anchor_a_long, lut.soft_anchor_b_long
            if a.dim() == 3:                       # block-diagonal, per-head slices
                H, T, NAP = a.shape
                ia = a.reshape(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
                ib = b.reshape(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
                d = (torch.gather(z, 2, ia) - torch.gather(z, 2, ib)).view(-1, H, T, NAP)
            else:                                  # shared code, [n_tables, NAP] anchors
                NAP = a.shape[-1]
                zz = z.reshape(z.shape[0], -1)
                d = zz[:, a] - zz[:, b]
            # keep the per-anchor structure: [tokens*H*T, NAP]
            rec['d'] = d.reshape(-1, NAP).detach().float().clone()
            # the compressed code itself, for the "what would wider margins take" question
            rec['z_std'] = z.std().item()

        h = ffn.register_forward_hook(hook)
        with torch.no_grad():
            m(x_ids)
        h.remove()
        per_block.append(rec)

    d_all = torch.cat([r['d'] for r in per_block])
    torch.save(dict(d=d_all,
                    per_block=[r['d'] for r in per_block],
                    z_std=[r['z_std'] for r in per_block],
                    nap=d_all.shape[-1],
                    config='sweep_s05_dout48_H4_tph256_c256_din32'), OUT)
    print(f'wrote {OUT}: d {tuple(d_all.shape)}  '
          f'|d| median {d_all.abs().median():.6f}  mean {d_all.abs().mean():.6f}')
    print('per-block compressed-code std:',
          '  '.join(f'{s:.4f}' for s in [r['z_std'] for r in per_block]))


if __name__ == '__main__':
    main()
