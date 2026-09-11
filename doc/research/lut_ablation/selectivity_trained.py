"""Within-token selectivity of a trained run's confidence score, on its own margins.

    python selectivity_trained.py <run_dir> [<run_dir> ...]

For each run: build the model from config.json, load checkpoint.pt, push 4 real val rows
(12 skipped, as the corrected eval does) through it, capture every LightMHL layer's margins, and
score them with the run's OWN confidence_form and confidence_gain via the library helper. Reports
per layer and overall: mean score, p25/p75 and p75/p25, CV, within-token CV (spread across tables
inside one (token, head), relative to the mean) and the fraction of scores below 1e-3.
CPU only, eval only; nothing is trained or written.
"""
import json
import os
import sys

import torch

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.join(FR, 'distill'))
import distill_ffn as D                                                   # noqa: E402
from model_build import build_model                                       # noqa: E402
from spiky.lutorch.fast_multi_head_lut import _confidence_score           # noqa: E402


def quant(t, ps=(0.25, 0.75)):
    g = torch.Generator().manual_seed(0)
    t = t[torch.randperm(t.numel(), generator=g)[:400000]] if t.numel() > 400000 else t
    return torch.quantile(t.double(), torch.tensor(ps, dtype=torch.float64)).tolist()


def stats(s, group):
    p25, p75 = quant(s)
    usable = (s.numel() // group) * group
    grp = s[:usable].view(-1, group)
    wcv = ((grp - grp.mean(dim=1, keepdim=True)).std() / grp.mean()).item()
    return dict(mean=s.mean().item(), p25=p25, p75=p75, ratio=p75 / max(p25, 1e-12),
                cv=(s.std() / s.mean()).item(), wcv=wcv, small=(s < 1e-3).float().mean().item())


def run(rd):
    rd = rd if os.path.isabs(rd) else os.path.join(FR, 'runs_corrected', rd)
    cfg = json.load(open(os.path.join(rd, 'config.json')))
    tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
    model = build_model(cfg, tok.get_vocab_size(), device='cpu')
    miss, unexp = model.load_state_dict(torch.load(os.path.join(rd, 'checkpoint.pt'), map_location='cpu'),
                                        strict=False)
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
    form, gain = cfg.get('lut_confidence_form', 'margin'), float(cfg.get('lut_confidence_gain', 1.0))
    gamma = cfg.get('lut_sharp_margin_gamma')                       # sharp_margin only, else None
    print('=' * 110)
    print(f'{os.path.basename(rd)}: form={form} gain={gain}' + (f' gamma={gamma}' if gamma is not None else '')
          + f' | missing keys {len(miss)} {sorted(set(k.split(".")[-1] for k in miss))}')
    print(f'   {"layer":<8} {"mean":>9} {"p25":>9} {"p75":>9} {"p75/p25":>9} {"CV":>7} {"within-tok CV":>14} {"frac<1e-3":>10}')
    allS = []
    for li, (lut, z) in enumerate(got):
        H, T, NAP = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
        a = lut.anchor_a.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        b = lut.anchor_b.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        d = (torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(z.shape[0], H, T, NAP).double()
        s = lut.confidence_score(d).reshape(-1)       # the layer's own form/gain/gamma/learned params; [tokens*H*T]
        allS.append(s)
        st = stats(s.float(), T)
        print(f'   L{li:<7} {st["mean"]:>9.4f} {st["p25"]:>9.4f} {st["p75"]:>9.4f} {st["ratio"]:>9.2f} '
              f'{st["cv"]:>7.3f} {st["wcv"]:>14.3f} {st["small"]:>10.4f}')
    st = stats(torch.cat(allS).float(), got[0][0].tables_per_head)
    print(f'   {"OVERALL":<8} {st["mean"]:>9.4f} {st["p25"]:>9.4f} {st["p75"]:>9.4f} {st["ratio"]:>9.2f} '
          f'{st["cv"]:>7.3f} {st["wcv"]:>14.3f} {st["small"]:>10.4f}')


if __name__ == '__main__':
    for r in sys.argv[1:]:
        run(r)
