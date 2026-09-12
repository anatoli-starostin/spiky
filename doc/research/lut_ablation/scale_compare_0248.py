"""Where does the confidence-score SCALE go when the learnable gain is removed? Read-only; CPU.

    python scale_compare_0248.py

Runs: exp_g_0193 (margin: g=0, beta=2, gamma=1 fixed), exp_g_0247 (g, beta, gamma learnable),
exp_g_0248 (g frozen at 0; beta, gamma learnable). All three share seed, data order and init.
At each saved checkpoint (4K, 8K, 12K, 16K), per layer, on 4 real val rows (skip 12):

  exp(g); ||decompress W||_F; ||tables||_F
  mean score E[s] (including exp(g)) and the SHAPE-ONLY mean E[s]/exp(g)
  RMS of the LUT head output y and of the FFN output decompress(y)
  gamma's scale elasticity  e_gamma = d log E[s] / d log gamma = gamma * E[s log P] / E[s]   (<= 0)

Reading: in 0247 g carried a downscale (exp(g) 0.77-1.0). If g is bookkeeping, 0248 reaches the same
FFN-output scale with larger decompress / tables and a gamma profile like 0247's. If gamma is now also
carrying scale, 0248's gamma departs from 0247's in the direction that reproduces 0247's score scale,
with |e_gamma| * |log(gamma_0248 / gamma_0247)| comparable to |g_0247|.
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RC = os.path.join(FR, 'runs_corrected')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.join(FR, 'distill'))
import distill_ffn as D                                                  # noqa: E402
from model_build import build_model                                      # noqa: E402

RUNS = {'0193': 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1',
        '0247': 'exp_g_0247_B16k_light_learnedmargin_tph128_seed1',
        '0248': 'exp_g_0248_B16k_light_learnedmargin_frozeng_tph128_seed1'}
CKPTS = [(4000, 'checkpoint_step4000.pt'), (8000, 'checkpoint_step8000.pt'),
         (12000, 'checkpoint_step12000.pt'), (16000, 'checkpoint.pt')]
L = 6

tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
V = tok.get_vocab_size()
cfgs = {k: json.load(open(os.path.join(RC, v, 'config.json'))) for k, v in RUNS.items()}
idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 16, cfgs['0193']['seq_len'], split='val',
                                                                  device='cpu'), 4, skip_rows=12)


def stats(run, fname):
    model = build_model(cfgs[run], V, device='cpu')
    miss, unexp = model.load_state_dict(torch.load(os.path.join(RC, RUNS[run], fname), map_location='cpu'),
                                        strict=False)
    model.eval()
    act = {}
    hooks = []
    for i, b in enumerate(model.blocks):
        def lut_hook(mod, inp, out, i=i):
            z = inp[0]
            H, T, NAP = mod.n_heads, mod.tables_per_head, mod.n_anchor_pairs
            a = mod.anchor_a.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
            bb = mod.anchor_b.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
            d = (torch.gather(z, 2, a) - torch.gather(z, 2, bb)).view(z.shape[0], H, T, NAP).double()
            s = mod.confidence_score(d)
            v = mod.learned_confidence_values() or {'g': 0.0, 'beta': 2.0, 'gamma': 1.0}
            logP = F.logsigmoid(v['beta'] * d.abs()).sum(-1)
            act[i] = {'score_mean': s.mean().item(), 'y_rms': out.pow(2).mean().sqrt().item(),
                      'e_gamma': (v['gamma'] * (s * logP).mean() / s.mean()).item(), **v}
        hooks.append(b.ffn.lut_light.register_forward_hook(lut_hook))
        hooks.append(b.ffn.register_forward_hook(
            lambda mod, inp, out, i=i: act[i].__setitem__('ffn_rms', out.pow(2).mean().sqrt().item())))
    with torch.no_grad():
        x = model.tok_emb(idx)
        for b in model.blocks:
            x = b(x, model.rope.cos, model.rope.sin)
    for h in hooks:
        h.remove()
    return [dict(dec=b.ffn.decompress.weight.norm().item(), tab=b.ffn.lut_light.tables.norm().item(), **act[i])
            for i, b in enumerate(model.blocks)], len(miss), len(unexp)


res = {}
for step, fname in CKPTS:
    for run in RUNS:
        res[(run, step)], nm, nu = stats(run, fname)
        print(f'loaded {run} @ {step}: missing {nm} unexpected {nu}', flush=True)

for step, _ in CKPTS:
    print('=' * 124)
    print(f'step {step}')
    print(f'   {"layer":<6}{"run":<6}{"exp(g)":>8}{"beta":>8}{"gamma":>8}{"||dec||":>9}{"||tab||":>9}{"E[s]":>9}'
          f'{"E[s]/e^g":>10}{"y rms":>9}{"ffn rms":>9}{"e_gamma":>9}')
    for i in range(L):
        for run in RUNS:
            r = res[(run, step)][i]
            print(f'   L{i:<5}{run:<6}{math.exp(r["g"]):>8.4f}{r["beta"]:>8.4f}{r["gamma"]:>8.4f}{r["dec"]:>9.4f}'
                  f'{r["tab"]:>9.4f}{r["score_mean"]:>9.4f}{r["score_mean"] / math.exp(r["g"]):>10.4f}'
                  f'{r["y_rms"]:>9.4f}{r["ffn_rms"]:>9.4f}{r["e_gamma"]:>9.4f}')
print('=' * 124)
print('16K summary per layer: 0248 vs 0247')
print(f'   {"layer":<6}{"g_0247":>9}{"gamma47":>9}{"gamma48":>9}{"e_g48":>8}{"scale via gamma":>17}'
      f'{"E[s] 48/47":>12}{"dec 48/47":>11}{"ffn 48/47":>11}{"ffn 48/93":>11}')
for i in range(L):
    a, b, c = res[('0247', 16000)][i], res[('0248', 16000)][i], res[('0193', 16000)][i]
    via = b['e_gamma'] * math.log(b['gamma'] / a['gamma'])          # log-scale change attributable to gamma
    print(f'   L{i:<5}{a["g"]:>+9.4f}{a["gamma"]:>9.4f}{b["gamma"]:>9.4f}{b["e_gamma"]:>8.3f}{via:>+17.4f}'
          f'{b["score_mean"] / a["score_mean"]:>12.4f}{b["dec"] / a["dec"]:>11.4f}{b["ffn_rms"] / a["ffn_rms"]:>11.4f}'
          f'{b["ffn_rms"] / c["ffn_rms"]:>11.4f}')
