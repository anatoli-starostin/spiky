"""Scale and selectivity of a proposed `tanh_margin` confidence form, on real margin vectors.

    margin        s = (sum_j m_j) * prod_j sigmoid(2 m_j)          (existing, the reference scale)
    tanh_margin   s = (sum_j m_j) * prod_j tanh(a m_j),  a = 1,2,3,4  (vanishes at every boundary)
    min_margin    s = (min_j m_j) * prod_j sigmoid(2 m_j)           (for reference)
    harmonic      s = (n / sum_j 1/m_j) * prod_j sigmoid(2 m_j)     (for reference)

Data, all nap = 8, m = |d|:
  * /tmp/margins_anchor.pt / _trained.pt  -- runs_corrected/dump_margins.py (sweep_s05, TPH=256), per block
  * exp_g_0193's own trained margins on 4 real val rows, per layer (TPH=128), computed on CPU

For every form: mean score, and the margin:form ratio (mean margin / mean form), per layer and
overall; for tanh_margin also p75/p25 and within-token CV (the across-tables spread inside one
token, grouped per head). Nothing is trained; no library code is used for the new forms.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

A_VALUES = (1.0, 2.0, 3.0, 4.0)
CACHES = [('init (sweep_s05)', '/tmp/margins_anchor.pt', 256),
          ('after 4K steps (sweep_s05)', '/tmp/margins_anchor_trained.pt', 256)]


def forms(m):
    n = m.shape[-1]
    sig = torch.exp(F.logsigmoid(2.0 * m).sum(dim=-1))
    out = {'margin': m.sum(dim=-1) * sig}
    for a in A_VALUES:
        out[f'tanh_margin a={a:g}'] = m.sum(dim=-1) * torch.tanh(a * m).prod(dim=-1)
    out['min_margin'] = m.min(dim=-1).values * sig
    out['harmonic'] = (n / (1.0 / m).sum(dim=-1)) * sig
    return out


def quant(t, ps=(0.25, 0.75)):
    g = torch.Generator().manual_seed(0)
    t = t[torch.randperm(t.numel(), generator=g)[:400000]] if t.numel() > 400000 else t
    return torch.quantile(t.double(), torch.tensor(ps, dtype=torch.float64)).tolist()


def within_cv(s, tph):
    usable = (s.numel() // tph) * tph
    grp = s[:usable].view(-1, tph)
    return ((grp - grp.mean(dim=1, keepdim=True)).std() / grp.mean()).item()


def report(tag, blocks, tph):
    print('=' * 118)
    print(f'{tag}: {sum(b.shape[0] for b in blocks):,} margin vectors, {len(blocks)} layers, TPH={tph}')
    names = list(forms(blocks[0][:10].abs().double()).keys())
    print('   mean score per layer / overall, and ratio margin:form (how many times smaller than margin)')
    print(f'   {"form":<18}' + ''.join(f'{"L" + str(i):>16}' for i in range(len(blocks))) + f'{"OVERALL":>18}')
    per = [forms(b.abs().double()) for b in blocks]
    allm = torch.cat([b.abs().double() for b in blocks])
    tot = forms(allm)
    for k in names:
        cells = ''.join(f'{p[k].mean().item():>8.4f} ({p["margin"].mean().item() / p[k].mean().item():>5.2f})'
                        for p in per)
        print(f'   {k:<18}{cells}{tot[k].mean().item():>9.4f} ({tot["margin"].mean().item() / tot[k].mean().item():>6.2f})')
    print(f'   selectivity (overall): {"form":<18} {"p75/p25":>8} {"within-token CV":>16} {"frac < 1e-3":>12}')
    for k in names:
        s = tot[k].float()
        p25, p75 = quant(s)
        print(f'   {"":<23}{k:<18} {p75 / max(p25, 1e-12):>8.2f} {within_cv(s, tph):>16.3f} '
              f'{(s < 1e-3).float().mean().item():>12.4f}')
    return {k: tot['margin'].mean().item() / tot[k].mean().item() for k in names}


def light_blocks():
    FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
    sys.path.insert(0, os.path.join(FR, 'tools'))
    sys.path.insert(0, os.path.join(FR, 'distill'))
    import distill_ffn as D
    from model_build import build_model
    rd = os.path.join(FR, 'runs_corrected', 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1')
    cfg = json.load(open(os.path.join(rd, 'config.json')))
    tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
    model = build_model(cfg, tok.get_vocab_size(), device='cpu')
    model.load_state_dict(torch.load(os.path.join(rd, 'checkpoint.pt'), map_location='cpu'), strict=False)
    model.eval()
    idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 16, cfg['seq_len'], split='val',
                                                                     device='cpu'), 4, skip_rows=12)
    got = []
    hooks = [b.ffn.lut_light.register_forward_hook(lambda mod, inp, out: got.append((mod, inp[0].detach())))
             for b in model.blocks]
    with torch.no_grad():
        x = model.tok_emb(idx)
        for b in model.blocks:
            x = b(x, model.rope.cos, model.rope.sin)
    for h in hooks:
        h.remove()
    blocks = []
    for lut, z in got:
        H, T, NAP = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
        a = lut.anchor_a.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        b = lut.anchor_b.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        blocks.append((torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(-1, NAP).float())
    return blocks, got[0][0].tables_per_head


if __name__ == '__main__':
    summary = {}
    for tag, path, tph in CACHES:
        if not os.path.exists(path):
            print(f'{path} missing -- run runs_corrected/dump_margins.py')
            continue
        blob = torch.load(path)
        summary[tag] = report(tag, blob['per_block'], tph)
    blocks, tph = light_blocks()
    summary['exp_g_0193 trained'] = report('exp_g_0193 trained (LightMHL margin, H4 tph128 nap8, 16K), 4 val rows',
                                           blocks, tph)
    print('=' * 118)
    print('OVERALL margin:form ratio by dataset')
    for tag, r in summary.items():
        print(f'   {tag:<30} ' + '  '.join(f'{k}={v:.2f}' for k, v in r.items() if k != 'margin'))
