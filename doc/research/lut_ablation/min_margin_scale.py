"""Scale and selectivity of a proposed `min_margin` confidence form, on real margin vectors.

    min_margin:  s = (min_j |u_j|) * prod_j sigmoid(2|u_j|)

Same analysis and the same data as runs_corrected/diag_confidence_forms.py:
  * /tmp/margins_anchor.pt          6,291,456 real nap-8 margin vectors at init
  * /tmp/margins_anchor_trained.pt  the same model after 4,000 steps
(both from runs_corrected/dump_margins.py, model sweep_s05_dout48_H4_tph256_c256_din32, TPH=256).
The three existing forms are recomputed first, so the published numbers
(bounded 0.0542 / margin 0.229 / bounded_norm 0.6838; p75/p25 2.06 / 3.03 / 1.09;
within-token CV 0.536 / 0.870 / 0.061) confirm the regenerated cache before min_margin is read.

Optionally also the trained Gen-3 standard layer, exp_g_0193 (LightMHL margin, H4 tph128 nap8,
16K), on real val tokens (--light).

Nothing is trained. No code in the library is touched: min_margin is computed here directly.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.fast_multi_head_lut import _confidence_score        # noqa: E402

CACHES = [('init', '/tmp/margins_anchor.pt', 256),
          ('after 4,000 steps', '/tmp/margins_anchor_trained.pt', 256)]
REF_BNORM = 0.6838   # the scale arms C and D were matched to (bounded_norm's mean at init)


def min_margin(d):
    m = d.abs()
    return m.min(dim=-1).values * torch.exp(F.logsigmoid(2.0 * m).sum(dim=-1))


def forms(d):
    return {'bounded': _confidence_score(d, 'bounded'),
            'margin': _confidence_score(d, 'margin'),
            'bounded_norm': _confidence_score(d, 'bounded_norm'),
            'min_margin': min_margin(d)}


def quant(t, ps=(0.25, 0.5, 0.75)):
    g = torch.Generator().manual_seed(0)
    t = t[torch.randperm(t.numel(), generator=g)[:400000]] if t.numel() > 400000 else t
    return torch.quantile(t.double(), torch.tensor(ps, dtype=torch.float64)).tolist()


def report(tag, d, tph):
    m = d.abs()
    N, nap = m.shape
    print('=' * 110)
    print(f'{tag}: {N:,} margin vectors, nap={nap}, |u| median {m.median():.4f} mean {m.mean():.4f}, '
          f'min_j|u_j| median {m.min(-1).values.median():.4f}')
    print(f'   {"form":<13} {"mean":>8} {"p25":>8} {"median":>8} {"p75":>8} {"p75/p25":>8} '
          f'{"CV":>6} {"within-tok CV":>14} {"gain->0.6838":>13} {"gain->margin":>13} {"gain->1":>8}')
    fs = forms(d)
    usable = (N // tph) * tph
    out = {}
    for k, s in fs.items():
        p25, med, p75 = quant(s)
        grp = s[:usable].view(-1, tph)
        within = grp - grp.mean(dim=1, keepdim=True)
        wcv = (within.std() / grp.mean()).item()
        mean = s.mean().item()
        out[k] = mean
        print(f'   {k:<13} {mean:>8.4f} {p25:>8.4f} {med:>8.4f} {p75:>8.4f} {p75 / max(p25, 1e-12):>8.2f} '
              f'{(s.std() / s.mean()).item():>6.3f} {wcv:>14.3f} {REF_BNORM / mean:>13.2f} '
              f'{fs["margin"].mean().item() / mean:>13.2f} {1.0 / mean:>8.2f}')
    zero = (fs['min_margin'] < 1e-3).float().mean().item()
    print(f'   min_margin: fraction of (token, table) scores below 1e-3: {zero:.4f}; '
          f'min_margin / margin ratio median {(fs["min_margin"] / fs["margin"]).median():.4f} '
          f'(upper bound 1/nap = {1 / nap:.4f})')
    return out


def nap_sweep(d):
    flat = d.abs().reshape(-1)
    g = torch.Generator().manual_seed(0)
    samp = flat[torch.randint(0, flat.numel(), (200000, 16), generator=g)]
    print(f'   nap-dependence (independent draws of the pooled empirical |u|):')
    print(f'   {"nap":>4} {"bounded":>10} {"bounded_norm":>13} {"margin":>10} {"min_margin":>11}')
    for nap in (1, 2, 4, 6, 8, 12, 16):
        f = forms(samp[:, :nap])
        print(f'   {nap:>4} {f["bounded"].mean():>10.5f} {f["bounded_norm"].mean():>13.5f} '
              f'{f["margin"].mean():>10.5f} {f["min_margin"].mean():>11.5f}')


def light_margins():
    """Margins of the trained Gen-3 standard layer (exp_g_0193) on real val tokens, all 6 layers."""
    import json
    FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
    sys.path.insert(0, os.path.join(FR, 'tools'))
    sys.path.insert(0, os.path.join(FR, 'distill'))
    import distill_ffn as D
    from model_build import build_model
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rd = os.path.join(FR, 'runs_corrected', 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1')
    cfg = json.load(open(os.path.join(rd, 'config.json')))
    tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
    model = build_model(cfg, tok.get_vocab_size(), device=dev)
    model.load_state_dict(torch.load(os.path.join(rd, 'checkpoint.pt'), map_location=dev), strict=False)
    model.eval()
    idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 16, cfg['seq_len'], split='val',
                                                                     device=dev), 4, skip_rows=12)
    got = []
    hooks = [b.ffn.lut_light.register_forward_hook(lambda mod, inp, out: got.append((mod, inp[0].detach())))
             for b in model.blocks]
    with torch.no_grad():
        x = model.tok_emb(idx)
        for b in model.blocks:
            x = b(x, model.rope.cos, model.rope.sin)
    for h in hooks:
        h.remove()
    ds = []
    for lut, z in got:
        H, T, NAP = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
        a = lut.anchor_a.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        b = lut.anchor_b.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        ds.append((torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(-1, NAP).float().cpu())
    return torch.cat(ds), got[0][0].tables_per_head


if __name__ == '__main__':
    for tag, path, tph in CACHES:
        if not os.path.exists(path):
            print(f'{path} missing -- run runs_corrected/dump_margins.py{" --trained" if "trained" in path else ""}')
            continue
        d = torch.load(path)['d']
        report(f'CACHE {tag}  ({os.path.basename(path)})', d, tph)
        if tag == 'init':
            nap_sweep(d)
    if '--light' in sys.argv:
        d, tph = light_margins()
        report('TRAINED Gen-3 standard layer exp_g_0193 (LightMHL margin, H4 tph128 nap8, 16K), 4 val rows', d, tph)
