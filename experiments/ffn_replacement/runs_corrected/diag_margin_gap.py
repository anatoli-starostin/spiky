"""STEP 1 — measure the margin GAP that sets the natural blend temperature.

WHY A GAP AT ALL. The blend weight is `w = softmax([0, -2 m_(1)/tau, ...])` over the winner
(cost 0) and the n-1 smallest-margin Hamming-1 flips. Its derivative w.r.t. the margins
carries the factor `w_k (1 - w_k) / tau`, which VANISHES AT BOTH ENDS: as tau -> 0 the
softmax goes one-hot and w(1-w) -> 0; as tau -> infinity the 1/tau prefactor kills it. The
routing gradient is therefore maximised when tau is matched to the scale of the cost gap
being discriminated. So the defensible init is tau ~ that gap.

WHAT THE GAP IS, precisely. For n=2 the blend competes exactly two candidates: the winner at
cost 0 and the nearest flip at cost 2*m_(1), where m_(1) = min_j |z[a_j] - z[b_j]| over the
nap anchor pairs of that (token, head, table). The cost DIFFERENCE is therefore
`2*m_(1) - 0`, i.e. the gap is m_(1) itself (up to the factor 2 carried in the exponent).
So: **delta_m := median m_(1)**, measured per layer on real val tokens.

For n=3 a third candidate enters at cost 2*m_(2), so the additional spacing that has to be
resolved is `m_(2) - m_(1)`. Reported alongside, because if that spacing is much smaller
than m_(1) then one temperature cannot resolve both competitions well and a per-candidate
or larger tau is the honest recommendation.

THE FACTOR OF 2, stated once because it is a live trap. The implementation uses
`exp(-2m/tau)`, so the effective temperature in units of the MARGIN is `tau_eff = tau/2`.
Matching `tau_eff = delta_m` therefore means setting **tau = 2 * delta_m**. This script
reports both so the resolved config value cannot be off by 2x.

Also reports cross-head and cross-table variation, since the implementation uses ONE scalar
per layer and that choice is only defensible if the spread within a layer is modest.

    python diag_margin_gap.py [run_dir]
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
import pyarrow.parquet as pq                                      # noqa: E402

RUN = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1
                   else 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = sys.argv[2] if len(sys.argv) > 2 else None

cfg = json.load(open(os.path.join(RUN, 'config.json')))
tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
val_shard = list_parquet_files()[-1]
texts = pq.read_table(val_shard).column('text').to_pylist()[:96]
ids = []
for t in texts:
    ids.extend(tok.encode(t))
    if len(ids) > 16 * 512:
        break
ids = torch.tensor(ids[:16 * 512], dtype=torch.long, device=DEV).view(16, 512)

model = build_model(cfg, tok.get_vocab_size(), device=DEV)
model.load_state_dict(torch.load(os.path.join(RUN, 'checkpoint.pt'), map_location=DEV),
                      strict=False)
model.eval()

caught = []


def hook(mod, inp, out, _s=caught):
    x = inp[0]
    H, T, K = mod.n_heads, mod.tables_per_head, mod.n_anchor_pairs
    B = x.shape[0]
    ia = mod.anchor_a.reshape(1, H, T * K).expand(B, H, T * K)
    ib = mod.anchor_b.reshape(1, H, T * K).expand(B, H, T * K)
    d = (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(B, H, T, K).abs()
    _s.append(d.detach().float().cpu())          # [B, H, T, K]


hs = [m.register_forward_hook(hook) for m in model.modules()
      if isinstance(m, LightMultiHeadLUT)]
with torch.no_grad():
    model(ids)
for h in hs:
    h.remove()

print(f'run   : {os.path.basename(RUN)}')
print(f'config: form={cfg.get("lut_confidence_form")} z_norm={cfg.get("lut_z_norm")} '
      f'nap={cfg["lut_n_anchor_pairs"]} tph={cfg["lut_tables_per_head"]} '
      f'H={cfg["lut_n_heads"]}')
print(f'tokens: {ids.numel():,} from {os.path.basename(val_shard)}\n')

print(f'{"layer":>5} {"med m_(1)":>10} {"mean m_(1)":>11} {"p25":>8} {"p75":>8} '
      f'{"med m2-m1":>10} | {"tau=2*dm":>9} {"w_nb@that":>10}')
per_layer = {}
for li, m in enumerate(caught):
    flat = m.reshape(-1, m.shape[-1])                       # [(B*H*T), K]
    s, _ = torch.sort(flat, dim=-1)
    m1, m2 = s[:, 0], s[:, 1]
    dm = m1.median().item()
    tau = 2.0 * dm
    # neighbour share at that tau: softmax([0, -2*m1/tau]) -> sigmoid(-2*m1/tau)
    w_nb = torch.sigmoid(-2.0 * m1 / tau).median().item()
    per_layer[li] = dict(median_m1=dm, mean_m1=m1.mean().item(),
                         p25=m1.quantile(.25).item(), p75=m1.quantile(.75).item(),
                         median_gap21=(m2 - m1).median().item(),
                         tau_matched=tau, w_neighbour_at_tau=w_nb)
    print(f'{li:>5} {dm:>10.5f} {m1.mean().item():>11.5f} '
          f'{m1.quantile(.25).item():>8.5f} {m1.quantile(.75).item():>8.5f} '
          f'{(m2 - m1).median().item():>10.5f} | {tau:>9.5f} {w_nb:>10.4f}')

allm1 = torch.cat([m.reshape(-1, m.shape[-1]).min(dim=-1).values for m in caught])
DM = allm1.median().item()
print(f'\nMODEL-WIDE  median m_(1) = delta_m = {DM:.6f}   -> matched tau = 2*delta_m = '
      f'{2*DM:.6f}')
print(f'            (tau_eff = tau/2 = {DM:.6f} in margin units, by construction)')

print('\n--- cross-head / cross-table spread WITHIN a layer (justifies one scalar or not) ---')
print(f'{"layer":>5} {"per-head median m_(1)":>44} {"max/min":>8} {"table-med CV":>13}')
for li, m in enumerate(caught):
    B, H, T, K = m.shape
    head_med = [m[:, h].reshape(-1, K).min(dim=-1).values.median().item() for h in range(H)]
    tbl_med = m.permute(2, 0, 1, 3).reshape(T, -1, K).min(dim=-1).values.median(dim=-1).values
    cv = (tbl_med.std() / tbl_med.mean()).item()
    print(f'{li:>5} {"  ".join(f"{v:.4f}" for v in head_med):>44} '
          f'{max(head_med)/max(min(head_med), 1e-12):>8.2f} {cv:>13.4f}')

if OUT:
    json.dump({'run': os.path.basename(RUN), 'delta_m_model': DM,
               'tau_matched_model': 2 * DM, 'per_layer': per_layer},
              open(OUT, 'w'), indent=2)
    print(f'\nwrote {OUT}')
