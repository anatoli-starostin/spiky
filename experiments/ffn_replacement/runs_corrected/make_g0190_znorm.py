"""exp_g_0190: LayerNorm on the compressed code before the lookup. One variable vs exp_g_0189.

THE HYPOTHESIS. Light's routing margins are thresholded against a code whose scale nothing
constrains: `z = compress(x)` goes straight into the lookup, and because the address is
`sign(d.detach())` no gradient pulls the scale anywhere. Measured, the margins drift by depth
in Light and not in Fast:

    median |d|   L0       L1     L2     L3     L4     L5
      LIGHT    0.00001  0.210  0.388  0.516  0.555  0.781    monotone
      FAST     0.274    0.712  0.676  0.674  0.663  0.603    flat after L0

and the gradient arriving at the FFN input is 13-16x smaller in Light at every layer while
its TABLE gradients are comparable -- i.e. the deficit is on the routing/input side. A
LayerNorm on z gives the margins a fixed scale to live on instead of a free one.

WHAT IT DOES AT INIT, measured rather than assumed: z std 0.37-0.41 -> 1.000, and median |d|
0.35-0.39 -> 0.96-0.98, a ~2.6x widening. The ADDRESSES are unchanged (0.0% of bits differ),
which is exactly right: with gain 1 and bias 0 the transform is z -> (z - mean)/std, the mean
cancels in d = z[a] - z[b], and the sign is therefore preserved. So at init this rescales the
margins and the gate without moving a single routing decision; only as the per-dimension
gains diverge from uniform can addresses change. A gentle intervention by construction.

Base is exp_g_0189 (1.207493), which already carries the tables-no-decay correction, so this
run isolates z_norm alone. Cost: 576 parameters (6 layers x 2 x 48), landing in the NODECAY
optimiser group (they are 1-D, verified by printing the group membership).

    python make_g0190_znorm.py
"""
import copy
import json
import os
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))

SRC = os.path.join(HERE, 'exp_g_0189_B16k_light_bnorm_tph128_nodecay_seed1')
RUN = 'exp_g_0190_B16k_light_bnorm_tph128_znorm_seed1'
EXPECT = 67_351_680 + 576

NOTE = (
    'ARM B AT 16K WITH A LAYERNORM ON THE COMPRESSED CODE (lut_z_norm=true) — '
    'LightMultiHeadLUT with the normalised confidence gate, copied from exp_g_0189 '
    '(1.207493) with lut_z_norm as the ONLY functional change, so it isolates z-normalisation '
    'on top of the tables-no-decay correction that 0189 already carries. THE HYPOTHESIS: '
    'Light thresholds its routing margins against a code whose scale nothing constrains — '
    'compress feeds the lookup directly, and since the address is sign(d.detach()) no gradient '
    'pulls that scale anywhere. Measured, the margins drift with depth in Light (median |d| '
    '0.00001 / 0.210 / 0.388 / 0.516 / 0.555 / 0.781, monotone) where Fast is flat (0.274 / '
    '0.712 / 0.676 / 0.674 / 0.663 / 0.603), and the gradient arriving at the FFN input is '
    '13-16x smaller in Light at every layer while its table gradients are comparable — the '
    'deficit is on the routing/input side, which is what this targets. IMPLEMENTATION: '
    'nn.LayerNorm(inner_in_dim) applied to z of shape [N, n_heads, 48], so each head\'s code '
    'is normalised independently over its own 48 dims with a shared learnable affine (gain 1, '
    'bias 0 at init); applied after compress and before the lookup, and before any '
    'inner_residual add (inner_residual is OFF here). Cost 576 parameters (6 layers x 2 x 48), '
    'total 67,352,256 vs 0189\'s 67,351,680; they are 1-D so they land in the NODECAY '
    'optimiser group, verified by printing the group membership rather than reasoning about '
    'it. MEASURED EFFECT AT INIT on real tokens: z std 0.37-0.41 -> 1.000 and median |d| '
    '0.35-0.39 -> 0.96-0.98, a ~2.6x widening of the margins, while 0.0% of address bits '
    'change — which is exactly right rather than suspicious: with gain 1 and bias 0 the '
    'transform is (z - mean)/std, the mean cancels in d = z[a] - z[b], and the sign is '
    'preserved. At init this rescales margins and the gate without moving any routing '
    'decision; only as the per-dimension gains diverge from uniform can addresses change. '
    'This run is also the FIRST with the per-layer ln1/ln2 logging in metrics.csv, so if '
    'layer 0 collapses again we will finally see WHEN — the open "never bootstrapped vs was '
    'alive then died" question that neither exp_n_0184 nor exp_g_0189 could answer. '
    'REFERENCES (corrected protocol, bs48 x 100, skip 12, 2,451,456 val tokens): exp_g_0189 '
    '1.207493 (the direct pair), exp_n_0185 1.206222, exp_n_0184 1.201075, exp_n_0186 '
    '1.208987, exp_n_0129 Fast gate-off 1.170961, vanilla dense exp_n_0135 1.165147 / '
    'exp_n_0176 1.161798 (vanilla seed spread 0.00335).')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

    base = json.load(open(os.path.join(SRC, 'config.json')))
    cfg = copy.deepcopy(base)
    cfg['lut_z_norm'] = True
    cfg['exp_name'] = RUN
    cfg['_arch_note'] = NOTE
    cfg['_sweep_tag'] = 'lookupffn-arm-b-16k-znorm'

    drift = [k for k in set(cfg) | set(base)
             if k not in ('exp_name', 'lut_z_norm', '_arch_note', '_sweep_tag')
             and cfg.get(k) != base.get(k)]
    if drift:
        print(f'*** STOP: unintended drift from exp_g_0189: {drift}')
        sys.exit(1)
    assert cfg.get('lut_inner_residual', False) is False
    assert cfg.get('lut_tables_no_decay') is True
    print('config diff vs exp_g_0189: + lut_z_norm=True (+ exp_name/_arch_note/_sweep_tag). '
          'No other field differs.')
    print(f'   inner_residual: {cfg.get("lut_inner_residual", "absent")} (OFF)   '
          f'tables_no_decay: {cfg["lut_tables_no_decay"]}   eval_every: {cfg["eval_every"]}')

    d = os.path.join(HERE, RUN)
    assert not os.path.exists(d), f'{d} exists -- never overwrite a prior run'
    os.makedirs(d)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    src_train, dst_train = os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py')
    shutil.copy(src_train, dst_train)
    assert open(src_train, 'rb').read() == open(dst_train, 'rb').read()
    trainer_src = open(dst_train).read()
    print(f'train.py byte-identical to train_fixed.py: OK')
    print(f'   carries the ln logging: {"ln2_norm_L" in trainer_src and "ln_stats" in trainer_src}')

    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()
    torch.manual_seed(cfg['random_seed'])
    m = build_model(cfg, vocab, device='cpu')
    tot = sum(p.numel() for p in m.parameters())
    ffn = m.blocks[0].ffn
    light = ffn.lut_light

    exempt = (FastMultiHeadLut, LightMultiHeadLUT)
    ids = {id(p) for mod in m.modules() if isinstance(mod, exempt)
           for p in mod.parameters(recurse=False)}
    names = {id(p): n for n, p in m.named_parameters()}
    nod = [names[id(p)] for p in m.parameters() if id(p) in ids or p.ndim < 2]

    checks = {
        'total params': (tot, EXPECT),
        'params added vs 0189': (tot - 67_351_680, 576),
        'z_norm in all blocks': ({b.ffn.z_norm is not None for b in m.blocks}, {True}),
        'z_norm normalised dim': (ffn.z_norm.normalized_shape, (48,)),
        'z_norm gain init': (float(ffn.z_norm.weight.mean()), 1.0),
        'z_norm bias init': (float(ffn.z_norm.bias.abs().max()), 0.0),
        'z_norm params in NODECAY': (len([n for n in nod if 'z_norm' in n]), 12),
        'inner_residual OFF': ({b.ffn.inner_residual for b in m.blocks}, {False}),
        'lut_light present': (light is not None, True),
        'tables_per_head': (light.tables_per_head, 128),
        'confidence_form': (light.confidence_form, 'bounded_norm'),
    }
    ok = True
    print(f'\n{RUN}')
    for name, (got, want) in checks.items():
        good = got == want
        ok &= good
        print(f'   {name:<30}{str(got)[:24]:>26}   expected {str(want)[:18]:<20}'
              f'{"OK" if good else "*** MISMATCH ***"}')

    with torch.no_grad():
        for b in m.blocks:
            b.ffn.decompress.weight.normal_(0, 2.3 / b.ffn.decompress.weight.numel() ** 0.5)
    loss = m(torch.randint(0, vocab, (2, 64))).float().mean()
    loss.backward()
    gz = ffn.z_norm.weight.grad
    print(f'\n   smoke (decompress driven to trained scale): loss {loss.item():.6g}  '
          f'tables grad {light.tables.grad.norm():.6g}  '
          f'compress grad {ffn.compress.weight.grad.norm():.6g}  '
          f'z_norm.weight grad {gz.norm():.6g}')
    ok &= bool(gz is not None and torch.isfinite(gz).all() and gz.norm() > 0)
    print(f'   z_norm receives gradient (it must, or the new params are dead): '
          f'{"YES" if gz.norm() > 0 else "*** NO ***"}')
    del m
    if not ok:
        print('\n*** STOP — not launching ***')
        sys.exit(1)
    print(f'\nwrote {d}/  — verified')


if __name__ == '__main__':
    main()
