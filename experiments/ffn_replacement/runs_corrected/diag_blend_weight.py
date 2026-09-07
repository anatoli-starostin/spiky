"""How aggressive is probe_soft_readout's n=2 blend, really?

The probe weights the 1-flip neighbour by u = exp(-2*m_min), where m_min is the SMALLEST of
the K anchor margins for that (token, table). Normalised, the neighbour's share is

    w1 = u / (1 + u) = exp(-2 m_min) / (1 + exp(-2 m_min)) = sigmoid(-2 m_min)

That is a per-(token, table) quantity and it decides whether "blend the top 2 cells" is a
gentle smoothing (w1 ~ 0.05) or a near coin-flip between two unrelated rows (w1 ~ 0.5).

It matters because m_min is an ORDER STATISTIC: even a model with large typical margins has
a small MINIMUM over K=8 draws, so the neighbour can carry real weight for most tokens even
when the routing is confident. If w1 is routinely large, a +0.02 regression at n=2 says the
blend was too blunt, NOT that smoothing is hopeless -- and the fix is a temperature on the
weight rather than abandoning the idea.

Reads real val tokens through the same held-out shard the scorer uses.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'tools')))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir                          # noqa: E402
from nanochat.tokenizer import RustBPETokenizer                   # noqa: E402
from nanochat.dataset import list_parquet_files                   # noqa: E402
from model_build import build_model                               # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RUNS = sys.argv[1:] or ['exp_g_0190_B16k_light_bnorm_tph128_znorm_seed1',
                        'exp_n_0192_repro0191_seed1']

# a small, fixed slab of real val tokens (same held-out shard the protocol scores on)
import pyarrow.parquet as pq                                       # noqa: E402
val_shard = list_parquet_files()[-1]
tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
texts = pq.read_table(val_shard).column('text').to_pylist()[:64]
ids = []
for t in texts:
    ids.extend(tok.encode(t))
    if len(ids) > 8 * 512:
        break
ids = torch.tensor(ids[:8 * 512], dtype=torch.long, device=DEV).view(8, 512)
print(f'val slab: {tuple(ids.shape)} tokens from {os.path.basename(val_shard)}')

for run in RUNS:
    d = os.path.join(HERE, run)
    ck = os.path.join(d, 'checkpoint.pt')
    if not os.path.exists(ck):
        print(f'\n### {run}: checkpoint absent, skipped')
        continue
    cfg = json.load(open(os.path.join(d, 'config.json')))
    model = build_model(cfg, tok.get_vocab_size(), device=DEV)
    model.load_state_dict(torch.load(ck, map_location=DEV), strict=False)
    model.eval()

    stats = []

    def hook(mod, inp, out, _s=stats):
        x = inp[0]
        H, T, K = mod.n_heads, mod.tables_per_head, mod.n_anchor_pairs
        B = x.shape[0]
        ia = mod.anchor_a.reshape(1, H, T * K).expand(B, H, T * K)
        ib = mod.anchor_b.reshape(1, H, T * K).expand(B, H, T * K)
        dd = (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(B, H, T, K).abs()
        _s.append(dd.detach().float().reshape(-1, K).cpu())

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, LightMultiHeadLUT)]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()

    print(f'\n### {run}')
    print(f'    form={cfg.get("lut_confidence_form")}  z_norm={cfg.get("lut_z_norm")}  '
          f'nap={cfg["lut_n_anchor_pairs"]}  tph={cfg["lut_tables_per_head"]}')
    print(f'    {"layer":>5} {"med |d|":>9} {"med m_min":>10} {"med w1":>8} {"p90 w1":>8} '
          f'{"frac w1>0.3":>12} {"frac w1>0.45":>13}')
    allm = []
    for li, m in enumerate(stats):
        mmin = m.min(dim=-1).values
        w1 = torch.sigmoid(-2.0 * mmin)
        allm.append(w1)
        print(f'    {li:>5} {m.median().item():>9.4f} {mmin.median().item():>10.4f} '
              f'{w1.median().item():>8.4f} {w1.quantile(0.9).item():>8.4f} '
              f'{(w1 > 0.3).float().mean().item():>12.4f} '
              f'{(w1 > 0.45).float().mean().item():>13.4f}')
    w = torch.cat(allm)
    print(f'    {"ALL":>5} {"":>9} {"":>10} {w.median().item():>8.4f} '
          f'{w.quantile(0.9).item():>8.4f} {(w > 0.3).float().mean().item():>12.4f} '
          f'{(w > 0.45).float().mean().item():>13.4f}')
    print(f'    mean neighbour share = {w.mean().item():.4f}  '
          f'-> the n=2 read is on average {100*w.mean().item():.1f}% "the other cell"')
    del model
    torch.cuda.empty_cache()
