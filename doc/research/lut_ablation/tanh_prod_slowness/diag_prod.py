"""Pin down WHY tanh_margin trains ~2x slower. Diagnosis only; nothing here is committed or used by a run.

Part A (ops, CUDA, d [6144, 4, 128, 8] = one micro-batch of one layer): forward and forward+backward
timings of the pieces of the tanh score vs margin's, and the aten ops autograd actually runs for
torch.prod's backward (CPU-side op names; this cage has no CUDA profiler activity).
Part B (causal A/B inside ONE process on the real model): time train.py's step with the stock
tanh_margin score, then with the tanh product's backward swapped for an equivalent-cost-free
formulation (exp(sum(log t)) -- same forward values away from exact zeros), same model and data.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_forms import FR, VARIANTS, sync_time                          # noqa: E402

DEV = 'cuda'


def ms(fn, n=30):
    for _ in range(3):
        fn()
    return sync_time(fn, n) * 1e3


g = torch.Generator(device=DEV).manual_seed(0)
d = torch.randn(6144, 4, 128, 8, device=DEV, generator=g) * 0.6
go = torch.randn(6144, 4, 128, device=DEV, generator=g)
m = d.abs()
t = torch.tanh(2.0 * m)
tl = t.clone().requires_grad_(True)
ml = m.clone().requires_grad_(True)


def fb(leaf, f):
    def run():
        (f(leaf) * go).sum().backward()
        leaf.grad = None
    return run


print('PART A: per call on [6144, 4, 128, 8]')
rows = {
    'prod(-1) forward': lambda: t.prod(-1),
    'cumprod(-1) forward': lambda: torch.cumprod(t, -1),
    'exp(log(t).sum(-1)) forward': lambda: torch.exp(torch.log(t).sum(-1)),
    'prod(-1) fwd+bwd  (leaf t)': fb(tl, lambda x: x.prod(-1)),
    'exp(log.sum) fwd+bwd (leaf t)': fb(tl, lambda x: torch.exp(torch.log(x).sum(-1))),
    'tanh(2m) elementwise fwd+bwd': fb(ml, lambda x: torch.tanh(2.0 * x).sum(-1)),
    'sum(m)*prod(tanh(2m)) fwd+bwd': fb(ml, lambda x: x.sum(-1) * torch.tanh(2.0 * x).prod(-1)),
    'margin sum(m)*exp(sum logsig) fwd+bwd': fb(ml, lambda x: x.sum(-1) * torch.exp(F.logsigmoid(2.0 * x).sum(-1))),
}
for k, f in rows.items():
    print(f'   {k:<40} {ms(f):8.2f} ms')

from torch.profiler import ProfilerActivity, profile                      # noqa: E402
out = tl.prod(-1)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    (out * go).sum().backward()
    torch.cuda.synchronize()
tl.grad = None
names = [(e.key, e.count, e.cpu_time_total / 1e3) for e in prof.key_averages()]
print('   aten ops in torch.prod backward (name, calls, CPU total ms incl. syncs):')
for k, c, tt in sorted(names, key=lambda x: -x[2]):
    if k.startswith('aten::') or 'Backward' in k:
        print(f'      {k:<45} {c:>4}  {tt:9.2f}')

print('PART B: real model train step, one process, stock vs swapped tanh product')
import json                                                              # noqa: E402
from model_build import build_model                                      # noqa: E402
import spiky.lutorch.light_multi_head_lut as L                          # noqa: E402
import spiky.lutorch.fast_multi_head_lut as FM                          # noqa: E402

cfg = json.load(open(os.path.join(FR, 'runs_corrected', 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1',
                                  'config.json')))
cfg.update(VARIANTS['tanh_margin'])
torch.manual_seed(0)
model = build_model(cfg, cfg['tokenizer_vocab_size'], device=DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
V, S, B = cfg['tokenizer_vocab_size'], cfg['seq_len'], cfg['device_batch_size']
xs = [torch.randint(0, V, (B, S), device=DEV, generator=g) for _ in range(4)]


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
    s = mm.sum(dim=-1) * torch.exp(torch.log(torch.tanh(FM.TANH_MARGIN_A * mm)).sum(dim=-1))
    return s if gain == 1.0 else s * gain


model.train()
for label, fn in (('stock  torch.prod', stock), ('swapped exp(sum log)', swapped), ('stock again', stock)):
    L._confidence_score = fn
    print(f'   {label:<22} train_step {ms(step, 20):8.1f} ms')
L._confidence_score = stock
