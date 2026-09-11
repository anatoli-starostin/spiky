"""Follow-up: is torch.prod's backward slow path triggered by EXACT ZERO factors, and does the real
tanh_margin model hit them? Diagnosis only.

A. synthetic [6144,4,128,8]: #exact zeros in diag_prod.py's tensor; prod fwd+bwd with no zeros vs ONE zero.
B. anchor pairs with a == b (a structurally zero margin) per layer, from the model's own buffers.
C. exact-zero margins d == 0 per layer: init model on random tokens (GPU, one micro-batch) and the
   TRAINED exp_g_0244 checkpoint on 4 real val rows (CPU).
D. real-model A/B in one process, NaN-safe: stock | log(t.clamp_min(1e-30)) swap | stock again,
   with the params checked finite after each phase.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_forms import FR, VARIANTS, sync_time                          # noqa: E402

sys.path.insert(0, os.path.join(FR, 'distill'))
import distill_ffn as D                                                  # noqa: E402
from model_build import build_model                                      # noqa: E402
import spiky.lutorch.fast_multi_head_lut as FM                          # noqa: E402
import spiky.lutorch.light_multi_head_lut as L                          # noqa: E402

DEV = 'cuda'


def ms(fn, n=20):
    for _ in range(3):
        fn()
    return sync_time(fn, n) * 1e3


print('A. synthetic')
g = torch.Generator(device=DEV).manual_seed(0)
d = torch.randn(6144, 4, 128, 8, device=DEV, generator=g) * 0.6
go = torch.randn(6144, 4, 128, device=DEV, generator=g)
t = torch.tanh(2.0 * d.abs())
print(f'   diag_prod tensor: exact zeros in d {(d == 0).sum().item()}, in tanh factors {(t == 0).sum().item()}')
t_nz = t.clamp_min(1e-20)
t_1z = t_nz.clone()
t_1z[100, 1, 7, 3] = 0.0
for label, base in (('no zero', t_nz), ('ONE zero', t_1z)):
    leaf = base.clone().requires_grad_(True)

    def fb():
        (leaf.prod(-1) * go).sum().backward()
        leaf.grad = None
    print(f'   prod fwd+bwd, {label:<9}: {ms(fb):7.2f} ms')


def hook_margins(model, run):
    got = []
    hs = [b.ffn.lut_light.register_forward_hook(lambda mod, inp, out: got.append((mod, inp[0].detach())))
          for b in model.blocks]
    with torch.no_grad():
        run()
    for h in hs:
        h.remove()
    rows = []
    for li, (lut, z) in enumerate(got):
        H, T, NAP = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
        a = lut.anchor_a.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        b = lut.anchor_b.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
        dd = (torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(z.shape[0], H, T, NAP)
        zero_tokens = (dd == 0).any(-1).any(-1).any(-1).float().mean().item()
        rows.append((li, int((lut.anchor_a == lut.anchor_b).sum()), int((dd == 0).sum()), dd.numel(), zero_tokens,
                     int((z == 0).sum()), z.numel()))
    return rows


def show(rows):
    for li, same, nz, n, ztok, zz, zn in rows:
        print(f'   L{li}: anchor pairs a==b {same:>3} | exact-zero margins {nz:>9,} / {n:,} ({nz / n:.2e}), '
              f'tokens with >=1 {ztok:.3f} | exact zeros in the code z {zz:,} / {zn:,}')


cfg = json.load(open(os.path.join(FR, 'runs_corrected', 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1',
                                  'config.json')))
cfg.update(VARIANTS['tanh_margin'])
V, S, B = cfg['tokenizer_vocab_size'], cfg['seq_len'], cfg['device_batch_size']
torch.manual_seed(0)
model = build_model(cfg, V, device=DEV)
xs = [torch.randint(0, V, (B, S), device=DEV, generator=g) for _ in range(4)]
print('B/C. init tanh_margin model, one random-token micro-batch (GPU)')
show(hook_margins(model, lambda: model(xs[0], xs[0])))

print('D. real-model A/B, one process')
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)


def step():
    opt.zero_grad(set_to_none=True)
    for x in xs:
        (model(x, x) / 4).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


stock = FM._confidence_score


def swapped(dd, form, gain=1.0, gamma=None):
    if form != 'tanh_margin':
        return stock(dd, form, gain, gamma)
    mm = dd.abs()
    s = mm.sum(dim=-1) * torch.exp(torch.log(torch.tanh(FM.TANH_MARGIN_A * mm).clamp_min(1e-30)).sum(dim=-1))
    return s if gain == 1.0 else s * gain


model.train()
for label, fn in (('stock torch.prod', stock), ('swapped clamp-log', swapped), ('stock again', stock)):
    L._confidence_score = fn
    tt = ms(step, 15)
    finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    print(f'   {label:<20} train_step {tt:7.1f} ms | params finite: {finite}')
L._confidence_score = stock
del model, opt
torch.cuda.empty_cache()

print('C. TRAINED exp_g_0244 checkpoint, 4 real val rows (skip 12), CPU')
rd = os.path.join(FR, 'runs_corrected', 'exp_g_0244_B16k_light_tanhmargin_a2_gain1_tph128_seed1')
cfg = json.load(open(os.path.join(rd, 'config.json')))
tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
m2 = build_model(cfg, tok.get_vocab_size(), device='cpu')
m2.load_state_dict(torch.load(os.path.join(rd, 'checkpoint.pt'), map_location='cpu'), strict=False)
m2.eval()
idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 16, cfg['seq_len'], split='val',
                                                                  device='cpu'), 4, skip_rows=12)


def run_trained():
    x = m2.tok_emb(idx)
    for b in m2.blocks:
        x = b(x, m2.rope.cos, m2.rope.sin)


show(hook_margins(m2, run_trained))
