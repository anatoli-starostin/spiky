"""Where did exp_g_0247's gain go in exp_g_0248 (g frozen)? Exact decomposition of the mean-score log-ratio
at 16K, per layer, on 4 real val rows (skip 12). Read-only; CPU.   python scale_decomposition_0248.py

With m47 / m48 the margins each trained model produces on the same tokens, p = (beta, gamma), and
E(m, p) the mean of (sum m) * prod sigmoid(beta m)^gamma (gain excluded):

  log E48 - log E47_with_gain
     = -g47                                        (0247's gain, absent in 0248)
     + [log E(m48, p48) - log E(m48, p47)]         (parameters, evaluated on 0248's margins)
         = beta part  [log E(m48, beta48, gamma47) - log E(m48, p47)]
         + gamma part [log E(m48, p48) - log E(m48, beta48, gamma47)]
     + [log E(m48, p47) - log E(m47, p47)]         (margin distribution, at 0247's parameters)

The terms sum exactly. Also: median |margin| per layer in both runs.
"""
import json
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

RUNS = {'0247': 'exp_g_0247_B16k_light_learnedmargin_tph128_seed1',
        '0248': 'exp_g_0248_B16k_light_learnedmargin_frozeng_tph128_seed1'}
tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
V = tok.get_vocab_size()
cfgs = {k: json.load(open(os.path.join(RC, v, 'config.json'))) for k, v in RUNS.items()}
idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 16, cfgs['0247']['seq_len'], split='val',
                                                                  device='cpu'), 4, skip_rows=12)


def margins_and_params(run):
    model = build_model(cfgs[run], V, device='cpu')
    model.load_state_dict(torch.load(os.path.join(RC, RUNS[run], 'checkpoint.pt'), map_location='cpu'), strict=True)
    model.eval()
    got = []
    hooks = [b.ffn.lut_light.register_forward_hook(lambda mod, inp, out: got.append((mod, inp[0].detach())))
             for b in model.blocks]
    with torch.no_grad():
        x = model.tok_emb(idx)
        for b in model.blocks:
            x = b(x, model.rope.cos, model.rope.sin)
    for h in hooks:
        h.remove()
    out = []
    for lut, z in got:
        H, T, NAP = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
        a = lut.anchor_a.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        b = lut.anchor_b.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        m = (torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(-1, NAP).abs().double()
        out.append((m, lut.learned_confidence_values()))
    return out


def logE(m, beta, gamma):
    return torch.log((m.sum(-1) * torch.exp(gamma * F.logsigmoid(beta * m).sum(-1))).mean()).item()


r47, r48 = margins_and_params('0247'), margins_and_params('0248')
print(f'{"layer":<6}{"total":>9}{"= -g47":>9}{"+ beta":>9}{"+ gamma":>9}{"+ margins":>11}{"check":>9}'
      f'{"med|m| 47":>11}{"med|m| 48":>11}')
for i, ((m47, p47), (m48, p48)) in enumerate(zip(r47, r48)):
    E47g = p47['g'] + logE(m47, p47['beta'], p47['gamma'])
    E48 = logE(m48, p48['beta'], p48['gamma'])
    total = E48 - E47g
    t_g = -p47['g']
    t_beta = logE(m48, p48['beta'], p47['gamma']) - logE(m48, p47['beta'], p47['gamma'])
    t_gamma = E48 - logE(m48, p48['beta'], p47['gamma'])
    t_marg = logE(m48, p47['beta'], p47['gamma']) - logE(m47, p47['beta'], p47['gamma'])
    print(f'L{i:<5}{total:>+9.4f}{t_g:>+9.4f}{t_beta:>+9.4f}{t_gamma:>+9.4f}{t_marg:>+11.4f}'
          f'{total - (t_g + t_beta + t_gamma + t_marg):>+9.1e}{m47.median().item():>11.4f}{m48.median().item():>11.4f}')
