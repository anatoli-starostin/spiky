"""Save the real LUT-layer input z (head 0) and its anchors, so cells_producer can build
a cells tensor from a real index distribution.

extract_real.py saved the top-1 indices and coefficients but not the activations they
came from, and the p2 cells need the margins, so this re-runs one real forward of
exp_n_0196 and keeps z itself.

NOTE on the scalars: exp_n_0196 was trained with confidence_form='margin', read_top_n=1
and no quant_mode, so it has no trained (tau, g, beta, gamma). The cells built from it
therefore use the p2_int8 preset's defaults -- tau = 0.1 and the learned_margin init
(g, beta, gamma) = (0, 2, 1), which is `margin` bit for bit -- applied to its REAL
margins. So the index distribution is real; the quantiser constants are the preset's.
"""
import json
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
NANOCHAT = os.path.expanduser('~/projects/nanochat')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
sys.path.insert(0, os.path.join(SPIKY, 'experiments/ffn_replacement/tools'))
sys.path.insert(0, NANOCHAT)

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402
from model_build import build_model                                # noqa: E402

RUN = os.path.join(SPIKY, 'experiments/ffn_replacement/runs_corrected',
                   'exp_n_0196_light_margin_znorm_nap8_tph256_seed1')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'artifacts', 'real_margins.pt')
DEVICE_BS, SEQ_LEN, HEAD = 48, 512, 0


@torch.no_grad()
def main():
    cfg = json.load(open(os.path.join(RUN, 'config.json')))
    model = build_model(cfg, cfg['tokenizer_vocab_size'], device='cuda')
    sd = torch.load(os.path.join(RUN, 'checkpoint.pt'), map_location='cuda')
    model.load_state_dict(sd.get('model', sd), strict=False)
    model.eval()
    name, lut = [(n, m) for n, m in model.named_modules()
                 if isinstance(m, LightMultiHeadLUT)][0]

    cap = {}
    h = lut.register_forward_hook(lambda m, i, o: cap.__setitem__('x', i[0].detach()))

    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DEVICE_BS, SEQ_LEN,
                                                            split='train', device='cuda')
    ids, _ = next(loader)
    model(ids)
    h.remove()

    H, din = lut.n_heads, lut.input_dim
    x = cap['x'].reshape(-1, H, din).float().contiguous()
    z = x[:, HEAD:HEAD + 1, :].contiguous()                     # [N, 1, din]
    a = lut.anchor_a[HEAD:HEAD + 1].contiguous()                # [1, T, nap]
    b = lut.anchor_b[HEAD:HEAD + 1].contiguous()
    print(f'layer {name}: heads {H}, din {din}, tables/head {lut.tables_per_head}, '
          f'nap {lut.n_anchor_pairs}')
    print(f'z head {HEAD}: {tuple(z.shape)} {z.dtype}   anchors {tuple(a.shape)}')
    print(f'z stats: mean {z.mean():.4f} std {z.std():.4f} '
          f'absmax {z.abs().max():.4f}')
    torch.save({'z': z.cpu(), 'anchor_a': a.cpu(), 'anchor_b': b.cpu(),
                'run': os.path.basename(RUN), 'layer': name, 'head': HEAD,
                'din': din, 'T': lut.tables_per_head, 'nap': lut.n_anchor_pairs},
               OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
