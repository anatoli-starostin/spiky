"""exp_n_0188: exp_n_0185 with the inner residual ON. Zero parameter delta, one variable.

WHY. On the exp_n_0184 checkpoint layer 0's ln2.weight had collapsed to zero (mean 0.000000,
norm 0.00386, against Fast's 0.894887 / 17.53908), driving the FFN input std to 0.00019, all
routing margins to ~0 and the bounded_norm gate to its 0.5 floor. Layer 0's FFN had become a
constant bias term -- 1,024 tables and 12.6M table parameters doing nothing token-dependent.
Fast, same geometry and position, shows healthy margins there (|d| median 0.273).

Read together with the other measurements, the picture is that Light's score-only gradient
could not bootstrap routing at layer 0, and the model's only way to make a non-routing layer
harmless was to switch it off. `inner_residual` gives it another option: with
decompress(lut(z) + z) the block can express a plain low-rank linear map even when the
lookup contributes nothing, so a layer whose routing has not started is useful rather than
dead -- and, importantly, keeps a live gradient path through compress, which is the same path
routing would need in order to start working later.

Two supporting measurements made this worth a run rather than a guess:
  * decompress amplifies the anchor-difference directions exactly as much as random ones
    (ratio 0.999-1.051 across layers), so the linear and routing objectives are
    geometrically indifferent -- no inherent tension for compress to resolve;
  * layers 1-5 are strongly nonlinear (held-out R^2 0.08-0.55, nonlinear residual 65-85% of
    the output norm), so the residual is not being added to something already linear.

Single variable against exp_n_0185 (1.206222): same seed, same sizing, ZERO parameter delta
-- the residual adds no parameters, so this run must report exactly 67,351,680.

    python make_b16k_innerres.py
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

SRC = os.path.join(HERE, 'exp_n_0185_B16k_light_bnorm_tph128_seed1')
RUN = 'exp_n_0188_B16k_light_bnorm_tph128_innerres_seed1'
EXPECT = 67_351_680

NOTE = (
    'ARM B AT 16K WITH THE INNER RESIDUAL ON (lut_inner_residual=true) — '
    'LightMultiHeadLUT with the normalised confidence gate (lut_impl="light", '
    'lut_forward_confidence=true, lut_confidence_form="bounded_norm"), copied from '
    'exp_n_0185 with lut_inner_residual as the ONLY functional change (asserted; any other '
    'drift aborts the build). The residual is applied inside CompressionMHL as '
    'decompress(lut(z) + z) in the inner 48-dim space, before decompress '
    '(compression_mhl.py:249-253 for the light multi-head branch, which is the path this '
    'config uses); it requires eff_in == eff_out, satisfied here at 48/48, and adds ZERO '
    'parameters — so this run must report exactly 67,351,680, identical to exp_n_0185. '
    'WHY: on the exp_n_0184 checkpoint layer 0\'s ln2.weight had collapsed to zero (mean '
    '0.000000, norm 0.00386, against Fast exp_n_0129\'s 0.894887 / 17.53908), driving the '
    'FFN input std to 0.00019, all routing margins to ~0 and the gate to its 0.5 floor; '
    'layer 0\'s FFN had become a constant bias term with 1,024 tables and 12.6M table '
    'parameters doing nothing token-dependent, while Fast\'s layer 0 at the same position '
    'shows healthy margins (|d| median 0.273). The reading is that Light\'s score-only '
    'gradient could not bootstrap routing at layer 0, and switching the layer off was the '
    'only way to make a non-routing layer harmless. The inner residual offers another '
    'option: the block can express a low-rank linear map when the lookup contributes '
    'nothing, keeping a live gradient path through compress — the same path routing needs to '
    'start working. Two measurements made this worth a run: decompress amplifies the '
    'anchor-difference directions exactly as much as random directions (ratio 0.999-1.051 '
    'per layer), so the linear and routing objectives are geometrically indifferent and '
    'compress is not being asked to trade one against the other; and layers 1-5 are strongly '
    'nonlinear (held-out R^2 0.08-0.55, nonlinear residual 65-85% of output norm), so the '
    'residual is not being added to something already linear. WHAT TO WATCH: whether layer '
    '0\'s ln2.weight survives instead of collapsing, and whether its margins stay alive. '
    'REFERENCES (corrected protocol, bs48 x 100, skip 12, 2,451,456 val tokens): exp_n_0185 '
    '1.206222 (the direct pair), exp_n_0184 tph256 1.201075, exp_n_0186 tph128/nap7 '
    '1.208987, exp_n_0129 Fast gate-off 1.170961, vanilla dense exp_n_0135 1.165147 / '
    'exp_n_0176 1.161798 (vanilla seed spread 0.00335).')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model

    base = json.load(open(os.path.join(SRC, 'config.json')))
    cfg = copy.deepcopy(base)
    cfg['lut_inner_residual'] = True
    cfg['exp_name'] = RUN
    cfg['_arch_note'] = NOTE
    cfg['_sweep_tag'] = 'lookupffn-arm-b-16k-innerres'

    drift = [k for k in set(cfg) | set(base)
             if k not in ('exp_name', 'lut_inner_residual', '_arch_note', '_sweep_tag')
             and cfg.get(k) != base.get(k)]
    if drift:
        print(f'*** STOP: unintended drift from exp_n_0185: {drift}')
        sys.exit(1)
    print(f'config diff vs exp_n_0185: + lut_inner_residual=True '
          f'(+ exp_name/_arch_note/_sweep_tag). No other field differs.')
    assert cfg['lut_inner_in_dim'] == cfg['lut_inner_out_dim'] == 48, 'eff_in must equal eff_out'

    d = os.path.join(HERE, RUN)
    assert not os.path.exists(d), f'{d} exists -- never overwrite a prior run'
    os.makedirs(d)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    src_train, dst_train = os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py')
    shutil.copy(src_train, dst_train)
    assert open(src_train, 'rb').read() == open(dst_train, 'rb').read()
    print('train.py byte-identical to train_fixed.py: OK')

    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()
    m = build_model(cfg, vocab, device='cpu')
    tot = sum(p.numel() for p in m.parameters())
    ffn = m.blocks[0].ffn
    light = ffn.lut_light
    checks = {
        'total params (== exp_n_0185)': (tot, EXPECT),
        'inner_residual ON in all blocks': ({b.ffn.inner_residual for b in m.blocks}, {True}),
        'light_multi_head_input (the path used)':
            ({b.ffn.light_multi_head_input for b in m.blocks}, {True}),
        'eff_in == eff_out': ((ffn.eff_in, ffn.eff_out), (48, 48)),
        'lut_light present': (light is not None, True),
        'lut_batched absent': (not hasattr(ffn, 'lut_batched'), True),
        'tables_per_head': (light.tables_per_head, 128),
        'n_anchor_pairs': (light.n_anchor_pairs, 8),
        'confidence_form': (light.confidence_form, 'bounded_norm'),
        'tables dtype fp32': (light.tables.dtype, torch.float32),
        'no temperature params': ([n for n, _ in m.named_parameters() if 'temp' in n], []),
    }
    ok = True
    print(f'\n{RUN}')
    for name, (got, want) in checks.items():
        good = got == want
        ok &= good
        print(f'   {name:<40}{str(got)[:22]:>24}   expected {str(want)[:18]:<20}'
              f'{"OK" if good else "*** MISMATCH ***"}')

    # Does the residual actually change the forward? It must be tested with decompress at a
    # NONZERO scale. model_build.py:144 zero-initialises decompress, and the residual enters
    # as decompress(lut(z) + z), so at init the whole effect is W_dec @ z = 0 -- the flag
    # looks inert while being perfectly wired. (Worth knowing in its own right: the residual
    # gives no output-level identity path at step 0. What it DOES change immediately is the
    # gradient to decompress, which now sees (y + z) instead of y.)
    with torch.no_grad():
        for b in m.blocks:
            b.ffn.decompress.weight.normal_(0, 2.3 / b.ffn.decompress.weight.numel() ** 0.5)
    torch.manual_seed(0)
    xs = torch.randn(8, 384)
    with torch.no_grad():
        y_res = ffn(xs)
        ffn.inner_residual = False
        y_no = ffn(xs)
        ffn.inner_residual = True
    delta = (y_res - y_no).norm() / y_no.norm().clamp_min(1e-30)
    print(f'\n   residual changes the FFN output (decompress at trained scale): '
          f'rel delta {delta:.5g}  {"OK" if delta > 1e-6 else "*** NO EFFECT ***"}')
    ok &= bool(delta > 1e-6)
    loss = m(torch.randint(0, vocab, (2, 64))).float().mean()
    loss.backward()
    gt, gc = light.tables.grad, ffn.compress.weight.grad
    print(f'   smoke fwd+bwd: loss {loss.item():.6g}  tables grad {gt.norm():.6g} {gt.dtype}  '
          f'compress grad {gc.norm():.6g}  finite={torch.isfinite(gt).all().item()}')
    ok &= bool(torch.isfinite(gt).all() and gt.norm() > 0 and gc.norm() > 0)
    del m
    if not ok:
        print('\n*** STOP — not launching ***')
        sys.exit(1)
    print(f'\nwrote {d}/  — verified')


if __name__ == '__main__':
    main()
