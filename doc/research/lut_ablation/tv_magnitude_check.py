"""Magnitude check of the Hamming-1 cell TV penalty (lut_cell_smoothness) against the main loss, BEFORE
training exp_g_0249 (= exp_g_0248 + TV weight 10). Nothing is trained or written.

    python tv_magnitude_check.py [--weight 10.0]

For (a) a FRESH exp_g_0248-config model (seed 1 init) and (b) exp_g_0248's TRAINED checkpoint, on ONE real
training step (4 micro-batches of 12 x 512 train tokens, exactly as train.py accumulates):
  CE loss (mean over micro-batches) and its gradient norm -- total, and on the LUT tables only
  TV penalty P = model.lut_tv_penalty() (mean over LightMHL layers of mean ||v_c - v_c'||^2 over Hamming-1
  pairs) and the norm of grad(weight * P) on the tables
  ratios: weight*P / CE, ||grad TV||_tables / ||grad CE||_tables, and TV's share of the COMBINED table
  gradient and of the global gradient norm that clip_grad_norm_(1.0) sees
Per layer: mean ||v_c - v_c'||^2 over Hamming-1 pairs, mean ||v_c||^2, and their ratio (how different
neighbouring cells are relative to cell magnitude).
"""
import json
import os
import sys

import torch

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RD = os.path.join(FR, 'runs_corrected', 'exp_g_0248_B16k_light_learnedmargin_frozeng_tph128_seed1')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.join(FR, 'distill'))
import distill_ffn as D                                                  # noqa: E402
from model_build import build_model                                      # noqa: E402

W = float(sys.argv[sys.argv.index('--weight') + 1]) if '--weight' in sys.argv else 10.0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = json.load(open(os.path.join(RD, 'config.json')))
tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
V = tok.get_vocab_size()
accum = cfg['total_batch_size'] // (cfg['device_batch_size'] * cfg['seq_len'])
loader = D.tokenizing_distributed_data_loader_bos_bestfit(tok, cfg['device_batch_size'], cfg['seq_len'],
                                                         split='train', device=DEV)
batches = [next(loader) for _ in range(accum)]


def norm(ps):
    gs = [p.grad for p in ps if p.grad is not None]
    return torch.sqrt(sum((g.double() ** 2).sum() for g in gs)).item() if gs else 0.0


def per_layer_cells(model):
    rows = []
    for li, b in enumerate(model.blocks):
        lut = b.ffn.lut_light
        nap, Dd = lut.n_anchor_pairs, lut.tables.shape[-1]
        t = lut.tables.detach().double().view(lut.n_tables, *([2] * nap), Dd)
        diff2 = sum((t.diff(dim=ax) ** 2).sum() for ax in range(1, nap + 1)) / (lut.n_tables * nap * (1 << (nap - 1)))
        cell2 = (lut.tables.detach().double() ** 2).sum(-1).mean()
        rows.append((li, diff2.item(), cell2.item()))
    return rows


def analyse(tag, model):
    model.train()
    tables = [b.ffn.lut_light.tables for b in model.blocks]
    allp = [p for p in model.parameters() if p.requires_grad]
    # CE gradient, accumulated exactly as train.py does
    model.zero_grad(set_to_none=True)
    ce = 0.0
    for x, y in batches:
        loss = model(x, y)
        (loss / accum).backward()
        ce += loss.item() / accum
    ce_tab, ce_all = norm(tables), norm(allp)
    g_ce = {id(p): (p.grad.clone() if p.grad is not None else None) for p in allp}
    # TV gradient alone (weight * P)
    model.zero_grad(set_to_none=True)
    P = model.lut_tv_penalty()
    (W * P).backward()
    tv_tab = norm(tables)
    g_tv = {id(p): (p.grad.clone() if p.grad is not None else None) for p in allp}
    # combined, as the trainer would present it to clip_grad_norm_
    comb_all = torch.sqrt(sum(((g_ce[i] if g_ce[i] is not None else 0) + (g_tv[i] if g_tv[i] is not None else 0)).double().pow(2).sum()
                              for i in g_ce if g_ce[i] is not None or g_tv[i] is not None)).item()
    comb_tab = torch.sqrt(sum(((g_ce[id(t)] if g_ce[id(t)] is not None else 0) + g_tv[id(t)]).double().pow(2).sum()
                              for t in tables)).item()
    cos = []
    for t in tables:
        a, b = g_ce[id(t)], g_tv[id(t)]
        if a is None or a.norm() == 0 or b.norm() == 0:
            cos.append(float('nan'))
        else:
            cos.append(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item())
    print('=' * 110)
    print(f'{tag}: weight {W}')
    print(f'   CE loss {ce:.4f} | TV penalty P {P.item():.4e} | weight*P {W * P.item():.4e} | weight*P / CE = {W * P.item() / ce:.3e}')
    print(f'   grad norms on TABLES: CE {ce_tab:.4e} | weight*TV {tv_tab:.4e} | ratio TV/CE = '
          f'{(tv_tab / ce_tab if ce_tab > 0 else float("inf")):.3e} | combined {comb_tab:.4e}')
    print(f'   global grad norm (what clip_grad_norm_(1.0) sees): CE only {ce_all:.4e} | CE+TV {comb_all:.4e} | '
          f'TV share of the global norm^2 = {tv_tab ** 2 / max(comb_all ** 2, 1e-30):.3e}')
    print(f'   cosine(grad CE, grad TV) per layer on the tables: {[round(c, 3) for c in cos]}')
    print('   per layer: mean ||v_c - v_c\'||^2 over Hamming-1 pairs | mean ||v_c||^2 | ratio')
    for li, d2, c2 in per_layer_cells(model):
        print(f'      L{li}: {d2:.4e} | {c2:.4e} | {d2 / c2 if c2 > 0 else float("nan"):.3f}')


torch.manual_seed(cfg['random_seed'])
fresh = build_model(cfg, V, device=DEV)
analyse('(a) FRESH INIT (exp_g_0248 config, seed 1)', fresh)
del fresh
trained = build_model(cfg, V, device=DEV)
trained.load_state_dict(torch.load(os.path.join(RD, 'checkpoint.pt'), map_location=DEV), strict=True)
analyse('(b) exp_g_0248 TRAINED (16K)', trained)
