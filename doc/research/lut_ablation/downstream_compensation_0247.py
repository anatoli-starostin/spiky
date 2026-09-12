"""Did the parameters DOWNSTREAM of the learned gain move inversely to exp(g) over exp_g_0247's training?
Read-only; CPU.   python downstream_compensation_0247.py

The learned_margin score enters the FFN linearly (score * table row, summed, then the decompress
Linear), so a global factor exp(g) can be undone by the tables or by decompress. Both runs share seed,
data order and init, so exp_g_0193 at the same step is the natural "no-g" reference. At each saved
checkpoint (4K, 8K, 12K, 16K) and per layer:

  weights   exp(g) (0247); ||decompress W||_F and ||tables||_F for both runs and their ratio 0247/0193
  activity  on 4 real val rows (skip 12): mean confidence score (INCLUDING exp(g)), RMS of the LUT head
            output y = sum_t s_t W_t[c_t], and RMS of the FFN output decompress(y); ratios 0247/0193

If g were pure bookkeeping, the downstream ratios would sit near exp(-g) and the FFN-output ratio near 1.
Caveats built into the reading: the two runs diverge after step 1 (gamma changes the score SHAPE and
its mean too, not just g), decompress is weight-decayed (0.1) while the tables are not, so a rescale is
representationally free but not free under the optimiser.
"""
import json
import math
import os
import sys

import torch

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RC = os.path.join(FR, 'runs_corrected')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.join(FR, 'distill'))
import distill_ffn as D                                                  # noqa: E402
from model_build import build_model                                      # noqa: E402

RUNS = {'0193': 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1',
        '0247': 'exp_g_0247_B16k_light_learnedmargin_tph128_seed1'}
CKPTS = [(4000, 'checkpoint_step4000.pt'), (8000, 'checkpoint_step8000.pt'),
         (12000, 'checkpoint_step12000.pt'), (16000, 'checkpoint.pt')]
L = 6

tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
V = tok.get_vocab_size()
cfgs = {k: json.load(open(os.path.join(RC, v, 'config.json'))) for k, v in RUNS.items()}
idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 16, cfgs['0193']['seq_len'], split='val',
                                                                  device='cpu'), 4, skip_rows=12)


def stats(run, fname):
    model = build_model(cfgs[run], V, device='cpu')
    miss, unexp = model.load_state_dict(torch.load(os.path.join(RC, RUNS[run], fname), map_location='cpu'),
                                        strict=False)
    model.eval()
    act = {}
    hooks = []
    for i, b in enumerate(model.blocks):
        lut = b.ffn.lut_light

        def lut_hook(mod, inp, out, i=i):
            z = inp[0]
            H, T, NAP = mod.n_heads, mod.tables_per_head, mod.n_anchor_pairs
            a = mod.anchor_a.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
            bb = mod.anchor_b.view(1, H, T * NAP).expand(z.shape[0], H, T * NAP)
            d = (torch.gather(z, 2, a) - torch.gather(z, 2, bb)).view(z.shape[0], H, T, NAP)
            act[i] = {'score_mean': mod.confidence_score(d).mean().item(), 'y_rms': out.pow(2).mean().sqrt().item()}

        hooks.append(lut.register_forward_hook(lut_hook))
        hooks.append(b.ffn.register_forward_hook(
            lambda mod, inp, out, i=i: act[i].__setitem__('ffn_rms', out.pow(2).mean().sqrt().item())))
    with torch.no_grad(), torch.enable_grad() if False else torch.no_grad():
        x = model.tok_emb(idx)
        for b in model.blocks:
            x = b(x, model.rope.cos, model.rope.sin)
    for h in hooks:
        h.remove()
    out = []
    for i, b in enumerate(model.blocks):
        lut = b.ffn.lut_light
        v = lut.learned_confidence_values()
        out.append(dict(g=(v['g'] if v else 0.0), dec=b.ffn.decompress.weight.norm().item(),
                        tab=lut.tables.norm().item(), **act[i]))
    return out, len(miss), len(unexp)


res = {}
for step, fname in CKPTS:
    for run in RUNS:
        res[(run, step)], nm, nu = stats(run, fname)
        print(f'loaded {run} @ {step}: missing {nm} unexpected {nu}', flush=True)

cols = ('exp(g)', 'dec 0247', 'dec 0193', 'dec ratio', 'exp(-g)', 'tab ratio', 'score ratio', 'y ratio', 'ffn ratio')
for step, _ in CKPTS:
    print('=' * 118)
    print(f'step {step}: ratios are 0247/0193 at the same step; score includes exp(g)')
    print(f'   {"layer":<6}' + ''.join(f'{c:>12}' for c in cols))
    for i in range(L):
        a, r = res[('0247', step)][i], res[('0193', step)][i]
        vals = (math.exp(a['g']), a['dec'], r['dec'], a['dec'] / r['dec'], math.exp(-a['g']), a['tab'] / r['tab'],
                a['score_mean'] / r['score_mean'], a['y_rms'] / r['y_rms'], a['ffn_rms'] / r['ffn_rms'])
        print(f'   L{i:<5}' + ''.join(f'{v:>12.4f}' for v in vals))
print('=' * 118)
print('across layers and checkpoints: correlation of log(dec ratio) with -g, and of log(score ratio) with g')
xs = [(-res[('0247', s)][i]['g'], math.log(res[('0247', s)][i]['dec'] / res[('0193', s)][i]['dec']),
       math.log(res[('0247', s)][i]['score_mean'] / res[('0193', s)][i]['score_mean']), res[('0247', s)][i]['g'])
      for s, _ in CKPTS for i in range(L)]
t = torch.tensor(xs)
print(f'   corr(-g, log dec ratio) = {torch.corrcoef(t[:, :2].T)[0, 1].item():+.3f}   '
      f'corr(g, log score ratio) = {torch.corrcoef(t[:, [3, 2]].T)[0, 1].item():+.3f}   (n = {len(xs)})')
