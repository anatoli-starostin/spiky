"""Run one real forward of a trained checkpoint and save the LUT layer's real
top-1 indices, real per-table coefficients, and int8-quantised tables.

Everything downstream (correctness, the repo-shape benchmark, the bucket statistics)
uses this artifact, so the model is built and run exactly once.

WHAT IS EXTRACTED, and why it is the module's own state rather than a reimplementation:
the script recomputes d / index / score from the module's registered anchor buffers and
its own `confidence_score`, then ASSERTS that sum_t score_t * tables[t, index_t]
reproduces the module's own forward output. If that assertion passes, the saved j and c
are the indices and coefficients the trained layer really uses.

QUANTISATION. The kernels take int8 tables, the checkpoint stores fp32. Tables are
quantised per table (one scale per table, s_t = max|T_t| / 127) and the scale is folded
into the coefficient, c'_t = c_t * s_t -- which is exactly the paper's per-table constant
coefficient folded into the table at load time (its Section 12.7). The quantisation error
against the fp32 tables is reported separately and is NOT part of kernel correctness:
kernel correctness is checked against the int8 values both paths actually read.
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
OUT = os.path.join(HERE, 'artifacts')
DEVICE_BS, SEQ_LEN = 48, 512          # 48 * 512 = 24,576 tokens = the real batch


def real_batch(vocab_size):
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    assert tok.get_vocab_size() == vocab_size, (tok.get_vocab_size(), vocab_size)
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, DEVICE_BS, SEQ_LEN, split='train', device='cuda')
    x, y = next(loader)
    return x


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(RUN, 'config.json')))
    model = build_model(cfg, cfg['tokenizer_vocab_size'], device='cuda')
    sd = torch.load(os.path.join(RUN, 'checkpoint.pt'), map_location='cuda')
    sd = sd.get('model', sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    name, lut = [(n, m) for n, m in model.named_modules()
                 if isinstance(m, LightMultiHeadLUT)][0]
    print(f'layer {name}: n_tables={lut.n_tables} table_size={lut.table_size} '
          f'output_dim={lut.output_dim} n_heads={lut.n_heads} '
          f'tables_per_head={lut.tables_per_head} forward_mode={lut.forward_mode} '
          f'confidence_form={lut.confidence_form} read_top_n={lut.read_top_n}')

    cap = {}

    def hook(mod, inp, out):
        cap['x'] = inp[0].detach()
        cap['y'] = out.detach()
    h = lut.register_forward_hook(hook)

    ids = real_batch(cfg['tokenizer_vocab_size'])
    print(f'real batch: {tuple(ids.shape)} tokens, dtype {ids.dtype}')
    model(ids)
    h.remove()

    x = cap['x'].reshape(-1, lut.n_heads * lut.input_dim).float().contiguous()
    y_mod = cap['y'].reshape(x.shape[0], lut.n_heads, lut.output_dim).float()
    B = x.shape[0]
    print(f'captured x {tuple(x.shape)}, module out {tuple(y_mod.shape)}')

    # ---- recompute the module's own addressing and score --------------------
    a, b = lut.native_anchor_a, lut.native_anchor_b            # [n_tables, NAP]
    d = x[:, a] - x[:, b]                                      # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    index = (bits * lut.powers.view(1, 1, -1)).sum(-1)          # [B, n_tables]
    score = lut.confidence_score(d).float()                     # [B, n_tables]
    assert index.max() < lut.table_size, index.max().item()

    tables = lut.tables.detach().float()                        # [n_tables, R, out]
    tix = torch.arange(lut.n_tables, device=x.device)
    rows = tables[tix.unsqueeze(0), index]                      # [B, n_tables, out]
    y_ref = (rows * score.unsqueeze(-1)).reshape(
        B, lut.n_heads, lut.tables_per_head, lut.output_dim).sum(2)
    err = (y_ref - y_mod).abs().max().item()
    scale = y_mod.abs().max().item()
    print(f'\nextraction check vs the module\'s own forward: max_abs={err:.3e} '
          f'rel={err/scale:.2e}   {"OK" if err/scale < 1e-4 else "FAIL"}')
    assert err / scale < 1e-4, 'extracted index/score do not reproduce the module output'

    # ---- per-table int8 quantisation, scale folded into the coefficient -----
    amax = tables.abs().amax(dim=(1, 2)).clamp_min(1e-12)       # [n_tables]
    s = amax / 127.0
    q = torch.round(tables / s.view(-1, 1, 1)).clamp(-127, 127).to(torch.int8)
    deq = q.float() * s.view(-1, 1, 1)
    qerr = (deq - tables).abs().max().item()
    print(f'int8 quantisation (per table, scale folded into c): '
          f'max |T - dequant(T)| = {qerr:.3e}  ({100*qerr/tables.abs().max().item():.3f}% of |T|max)')

    c_folded = (score * s.view(1, -1)).contiguous()

    art = {
        'run': os.path.basename(RUN), 'layer': name,
        'n_heads': lut.n_heads, 'tables_per_head': lut.tables_per_head,
        'table_size': lut.table_size, 'output_dim': lut.output_dim,
        'B': B,
        'j': index.to(torch.uint8).cpu(),                       # [B, n_tables]
        'c': c_folded.cpu(),                                    # [B, n_tables], scale folded
        'T_int8': q.cpu(),                                      # [n_tables, R, out]
        'table_scale': s.cpu(),
        'y_module': y_mod.cpu(),
        'quant_max_abs_err': qerr,
        'extract_rel_err': err / scale,
    }
    p = os.path.join(OUT, 'real_layer.pt')
    torch.save(art, p)
    print(f'\nsaved {p}')
    print(f'  j {tuple(art["j"].shape)} uint8, c {tuple(art["c"].shape)} fp32, '
          f'T_int8 {tuple(art["T_int8"].shape)}')

    # a quick look at how non-uniform the real index distribution is
    for hh in range(lut.n_heads):
        jh = index[:, hh * lut.tables_per_head:(hh + 1) * lut.tables_per_head]
        cnt = torch.zeros(lut.tables_per_head, lut.table_size, device=x.device)
        cnt.scatter_add_(1, jh.t().contiguous(), torch.ones_like(jh.t(), dtype=torch.float))
        p_ = cnt / cnt.sum(1, keepdim=True)
        pr = 1.0 / (p_ ** 2).sum(1)
        print(f'  head {hh}: participation ratio over the {lut.table_size} rows '
              f'mean {pr.mean():.1f}  min {pr.min():.1f}  max {pr.max():.1f}')


if __name__ == '__main__':
    main()
