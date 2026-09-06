"""Is Light's layer-0 collapse the tail of a global pathology (H1) or layer-0-specific (H2)?

Light (exp_n_0184) vs Fast (exp_n_0129) at identical geometry. Fast does NOT collapse, so it
is the control that makes every number interpretable: a quantity that looks extreme in Light's
layer 0 only matters if Fast's layer 0 looks different.

H1 GLOBAL GRADIENT PATHOLOGY -> the quantities trend monotonically across layers 0..5 with
   layer 0 as the tail, and the trend is present in Light but not (or less) in Fast.
H2 LAYER-0-SPECIFIC INPUT -> at layer 0 the residual stream is nearly a function of TOKEN ID
   alone (little contextual variation), so routing has no contextual signal to exploit. Then
   the input statistics should look the same in BOTH models -- and if only Light collapses,
   H2 alone cannot be the explanation.

The H2 measurement is a variance decomposition on the residual stream ENTERING ln2, not on
ln2's output: Light's layer-0 ln2 gain has collapsed to ~0, so its output is degenerate by
construction and would beg the question. Grouping tokens by id and comparing between-group to
total variance gives "how much of the representation is predictable from the token id alone".

CPU only.

    python diag_layer_trend.py
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

from nanochat.common import get_base_dir                                  # noqa: E402
from nanochat.tokenizer import RustBPETokenizer                           # noqa: E402
from nanochat.dataloader import (                                         # noqa: E402
    tokenizing_distributed_data_loader_bos_bestfit)
from model_build import build_model                                       # noqa: E402
from spiky.lutorch.fast_multi_head_lut import _confidence_score           # noqa: E402

MODELS = [('LIGHT', 'exp_n_0184_B16k_light_bnorm_seed1'),
          ('FAST ', 'exp_n_0129_grid_H4d48_nap8_tph256')]
torch.manual_seed(0)


def eff_rank(X):
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc.double())
    s2 = s.pow(2)
    return (s2.sum().pow(2) / s2.pow(2).sum().clamp_min(1e-300)).item()


def token_id_r2(X, ids):
    """Fraction of the representation's variance explained by token IDENTITY alone.

    Between-group variance over total variance, groups = token ids. 1.0 means the vector is
    a pure function of the token; 0 means identity tells you nothing beyond the mean.
    """
    X = X.double()
    total = (X - X.mean(0, keepdim=True)).pow(2).sum()
    uniq, inv = torch.unique(ids, return_inverse=True)
    sums = torch.zeros(len(uniq), X.shape[1], dtype=X.dtype).index_add_(0, inv, X)
    cnt = torch.zeros(len(uniq), dtype=X.dtype).index_add_(
        0, inv, torch.ones(len(ids), dtype=X.dtype))
    means = sums / cnt.unsqueeze(1)
    between = (cnt.unsqueeze(1) * (means - X.mean(0, keepdim=True)).pow(2)).sum()
    return (between / total.clamp_min(1e-300)).item(), len(uniq)


def main():
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    ld = tokenizing_distributed_data_loader_bos_bestfit(tok, 48, 512, split='val',
                                                        device='cpu')
    x_all, _ = next(iter(ld))
    x_ids = x_all[12:16].clone()
    flat_ids = x_ids.reshape(-1)
    print(f'{x_ids.numel():,} tokens, rows [12,16) of the corrected eval slice\n')

    store = {}
    for label, run in MODELS:
        D = os.path.join(RC, run)
        cfg = json.load(open(os.path.join(D, 'config.json')))
        sd = torch.load(os.path.join(D, 'checkpoint.pt'), map_location='cpu')
        torch.manual_seed(cfg['random_seed'])
        m0 = build_model(cfg, tok.get_vocab_size(), device='cpu')
        init_c = [b.ffn.compress.weight.detach().norm().item() for b in m0.blocks]
        del m0
        m = build_model(cfg, tok.get_vocab_size(), device='cpu')
        m.load_state_dict(sd, strict=False)
        m.eval()

        rec = {}
        hooks = []
        for li, blk in enumerate(m.blocks):
            def mk_ln(li):
                def h(mod, inp, out, li=li):
                    rec.setdefault(li, {})['resid'] = inp[0].detach()
                    rec[li]['lnout'] = out.detach()
                return h

            def mk_ffn(li, ffn):
                def h(mod, inp, out, li=li, ffn=ffn):
                    z = ffn.compress(inp[0]).detach()
                    lut = ffn.lut_light if hasattr(ffn, 'lut_light') else ffn.lut_batched
                    H, din = ffn.n_heads, ffn.inner_in_dim
                    NAP, T = lut.n_anchor_pairs, lut.tables_per_head
                    if hasattr(ffn, 'lut_light'):
                        zz = z.view(z.shape[0], H, din)
                        ia = lut.anchor_a.reshape(1, H, T*NAP).expand(z.shape[0], H, T*NAP)
                        ib = lut.anchor_b.reshape(1, H, T*NAP).expand(z.shape[0], H, T*NAP)
                        d = (torch.gather(zz, 2, ia) - torch.gather(zz, 2, ib)).view(
                            z.shape[0], H, T, NAP)
                    else:
                        a, b_ = lut.soft_anchor_a_long, lut.soft_anchor_b_long
                        d = (z[:, a.reshape(-1)] - z[:, b_.reshape(-1)]).view(
                            z.shape[0], H, T, NAP)
                    rec[li].update(z=z, d=d.abs(), out=out.detach(),
                                   gate=_confidence_score(d, 'bounded_norm', 1.0))
                return h
            hooks.append(blk.ln2.register_forward_hook(mk_ln(li)))
            hooks.append(blk.ffn.register_forward_hook(mk_ffn(li, blk.ffn)))
        with torch.no_grad():
            m(x_ids)
        for h in hooks:
            h.remove()
        store[label] = (m, rec, init_c, sd)

    print('=' * 118)
    print('A. THE FULL TREND, ALL SIX LAYERS, BOTH MODELS')
    print('=' * 118)
    print(f'{"":<7}{"":<3}{"ln2.w mean":>12}{"ln2.w norm":>12}{"|Wcomp|":>10}'
          f'{"/init":>8}{"|Wdec|":>9}{"ln2out std":>12}{"z per-dim std":>15}'
          f'{"gate med":>11}{"|d| med":>10}{"FFNout/resid":>14}')
    for label, _ in MODELS:
        m, rec, init_c, sd = store[label]
        for li in range(6):
            r = rec[li]
            w2 = sd[f'blocks.{li}.ln2.weight']
            wc = m.blocks[li].ffn.compress.weight.detach().norm().item()
            wd = m.blocks[li].ffn.decompress.weight.detach().norm().item()
            zs = r['z'].view(r['z'].shape[0], m.blocks[li].ffn.n_heads, -1).std(0).median()
            ratio = r['out'].norm(dim=-1).median() / r['resid'].norm(dim=-1).median()
            print(f'{label:<7}{li:<3}{w2.mean():>12.6f}{w2.norm():>12.5f}{wc:>10.4f}'
                  f'{wc/init_c[li]:>8.2f}{wd:>9.4f}{r["lnout"].std():>12.5f}'
                  f'{zs:>15.6f}{r["gate"].median():>11.5f}{r["d"].median():>10.5f}'
                  f'{ratio:>14.4f}')
        print()

    print('=' * 118)
    print('B. H2 TEST — is the residual stream entering ln2 just a function of TOKEN ID?')
    print('   (measured PRE-ln2, so Light\'s collapsed gain cannot beg the question)')
    print('=' * 118)
    print(f'{"":<7}{"":<3}{"token-id R^2":>14}{"eff rank of resid":>20}'
          f'{"resid std":>12}   (R^2 -> 1 means no contextual information)')
    for label, _ in MODELS:
        m, rec, _, _ = store[label]
        for li in range(6):
            # ln2 sees [B, T, C]; the FFN reshapes to [B*T, C]. Flatten to match flat_ids.
            X = rec[li]['resid'].reshape(-1, rec[li]['resid'].shape[-1])
            r2, nuniq = token_id_r2(X, flat_ids)
            print(f'{label:<7}{li:<3}{r2:>14.4f}{eff_rank(X):>20.2f}{X.std():>12.5f}')
        print()
    print(f'   ({nuniq:,} distinct token ids among {len(flat_ids):,} positions)')


if __name__ == '__main__':
    main()
