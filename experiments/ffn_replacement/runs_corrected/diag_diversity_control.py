"""Control: do the diversity statistics distinguish a TRAINED model from a random one?

High effective rank and low mean-dominance are only evidence of healthy structure if an
UNTRAINED model does not score the same. Random 256x48 matrices are full-rank almost surely,
so this control is what decides whether the metrics mean anything.
"""
import json
import os
import sys

import torch

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RC = os.path.join(FR, 'runs_corrected')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))

D = os.path.join(RC, 'exp_n_0184_B16k_light_bnorm_seed1')
cfg = json.load(open(os.path.join(D, 'config.json')))
sd = torch.load(os.path.join(D, 'checkpoint.pt'), map_location='cpu')
keys = [k for k in sd if k.endswith('lut_light.tables')]

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from model_build import build_model

vocab = RustBPETokenizer.from_directory(
    os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()
torch.manual_seed(cfg['random_seed'])
m0 = build_model(cfg, vocab, device='cpu')          # untrained, same seeds
init = {f'blocks.{i}.ffn.lut_light.tables': b.ffn.lut_light.tables.detach()
        for i, b in enumerate(m0.blocks)}


def stats(W):
    W = W.float()
    s = torch.linalg.svdvals(W)
    s2 = s.pow(2)
    pr = s2.sum(-1).pow(2) / s2.pow(2).sum(-1).clamp_min(1e-30)
    mu = W.mean(dim=1)
    dom = mu.pow(2).sum(-1) / (W - mu.unsqueeze(1)).pow(2).sum(-1).mean(-1).clamp_min(1e-30)
    return pr, dom, W.std()


print(f"{'':<10}{'participation ratio':>34}{'mean-row dominance':>28}{'table std':>14}")
print(f"{'layer':<10}{'trained':>13}{'random init':>13}{'':>8}{'trained':>10}"
      f"{'random':>10}{'':>8}{'trained':>7}{'random':>7}")
for i, k in enumerate(keys):
    pr_t, dom_t, sd_t = stats(sd[k])
    pr_r, dom_r, sd_r = stats(init[k])
    print(f'   {i:<7}{pr_t.median():>13.2f}{pr_r.median():>13.2f}{"":>8}'
          f'{dom_t.median():>10.5f}{dom_r.median():>10.5f}{"":>8}'
          f'{sd_t:>7.4f}{sd_r:>7.5f}')

pt = torch.cat([stats(sd[k])[0] for k in keys])
pr_ = torch.cat([stats(init[k])[0] for k in keys])
print(f'\n   participation ratio, all 6,144 tables:')
print(f'      trained     med {pt.median():.2f}   IQR [{pt.quantile(.25):.2f}, {pt.quantile(.75):.2f}]')
print(f'      random init med {pr_.median():.2f}   IQR [{pr_.quantile(.25):.2f}, {pr_.quantile(.75):.2f}]')
print(f'\n   VERDICT: if these are close, effective rank does NOT distinguish trained')
print(f'   structure from noise, and a high value is not evidence of healthy diversity.')

# how far did the tables actually move from init?
print('\n   how far did training move the tables?')
for i, k in enumerate(keys):
    a, b = sd[k].float(), init[k].float()
    rel = (a - b).norm() / b.norm()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    print(f'      layer {i}: ||W_trained - W_init|| / ||W_init|| = {rel:8.2f}   '
          f'cos(trained, init) = {cos:+.5f}   ||W_t||/||W_i|| = {a.norm()/b.norm():.2f}')
