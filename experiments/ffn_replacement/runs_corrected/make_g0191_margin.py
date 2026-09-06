"""exp_g_0191: the LookupFFN score on Light. One variable vs exp_g_0190.

WHAT THIS IS. The LookupFFN paper's differentiable hash score is, to machine precision,
our `margin` confidence_form. Their kernel computes

    score = (sum_j m_j) / prod_j (1 + exp(-2 m_j)),      m = |z|

and since sigmoid(2m) = 1/(1 + exp(-2m)) that is exactly our

    margin: score = (sum_j m_j) * prod_j sigmoid(2 m_j)

Checked numerically against research/lookupffn/lookup_ffn.py (nucstar's reference
implementation of arXiv:2403.07221) at |z| scales 0.01 / 0.1 / 1 / 3 / 10: max relative
difference 1.9e-15. Their bit-packed address rule is identical to ours as well. So this
run needs NO new code -- it is a config change -- and it is the LookupFFN mechanism on
our Light path.

WHY IT MIGHT MATTER. Our diagnosis is a routing-side gradient deficit: Light delivers
13-16x less gradient into the FFN input than Fast at every layer while table gradients
are comparable. The two score forms differ exactly in how that gradient behaves as the
routing margins grow:

    |z| per coord   margin (theirs)     d/dm   bounded_norm (ours)     d/dm
             0.10          0.006683   0.014370            0.549834   0.061879
             0.40          0.164367   0.153280            0.689974   0.053477
             0.80          1.469750   0.723430            0.832018   0.034941
             1.50          8.135261   1.449582            0.952574   0.011294
             3.00         23.529345   1.096748            0.997527   0.000617
             6.00         47.997641   1.000541            0.999994   0.000002

bounded_norm saturates and its routing gradient collapses; margin's sum_j|z_j| factor
holds d/dm near 1 indefinitely. Measured on our own module at init, changing only the
form string gives 8.9x more gradient into compress and grad_x (1.68e-05 vs 1.88e-06)
with table gradients within 6% -- the same signature as the Fast-vs-Light gap.

THE CONFOUND, stated plainly. This stacks margin ON TOP OF z_norm; it does NOT isolate
margin from z_norm. If margin wins, margin-without-z_norm is the follow-up needed to
deconfound. The reason to stack rather than isolate is that they are plausibly
complementary: z_norm pins |z| to std ~1, which is precisely where margin still has a
healthy d/dm (~0.15-0.7) and where bounded_norm has already begun to saturate. That is
a hypothesis for why the pair may work together, not a claim that it will.

PRIOR: `margin` has been run once, as arm D at 4k -- but on FAST, never on Light. It
scored 1.432430 against baseline 1.434572 (-0.002142), the only arm of the four that
went below baseline, though inside the 0.0096 4k seed sd. Light + margin is new.

    python make_g0191_margin.py
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

SRC = os.path.join(HERE, 'exp_g_0190_B16k_light_bnorm_tph128_znorm_seed1')
RUN = 'exp_g_0191_B16k_light_margin_tph128_znorm_seed1'
EXPECT = 67_352_256          # IDENTICAL to exp_g_0190: the score form adds no parameters

NOTE = (
    'THE LOOKUPFFN SCORE ON LIGHT (lut_confidence_form=margin) — copied from exp_g_0190 '
    '(1.203936) with the confidence form as the ONLY change, so it is a single variable '
    'against our current best. WHY THIS IS THE LOOKUPFFN MECHANISM: the paper '
    '(arXiv:2403.07221) computes score = (sum_j m_j) / prod_j (1 + exp(-2 m_j)) with '
    'm = |z|, and since sigmoid(2m) = 1/(1 + exp(-2m)) that is exactly our margin form, '
    '(sum_j m_j) * prod_j sigmoid(2 m_j). Verified numerically against nucstar\'s '
    'reference implementation (research/lookupffn/lookup_ffn.py on branch '
    'research/lookupffn) at |z| scales 0.01/0.1/1/3/10: max relative difference 1.9e-15. '
    'The bit-packed sign address is identical too. So no new code was needed. '
    'WHY IT MIGHT HELP: our diagnosis is a routing-side gradient deficit — Light delivers '
    '13-16x less gradient into the FFN input than Fast at every layer while table '
    'gradients are comparable. bounded_norm SATURATES as margins grow (d/dm falls '
    '0.0619 -> 0.0349 -> 0.0113 -> 0.0006 at |z| = 0.1/0.8/1.5/3.0) whereas margin\'s '
    'sum_j|z_j| factor holds d/dm near 1 indefinitely; and we measured Light\'s margins '
    'GROWING with depth (median |d| 0.00001/0.210/0.388/0.516/0.555/0.781), i.e. our gate '
    'switches itself off exactly where the deficit was measured. On our own module at '
    'init, changing only this string gives 8.9x more gradient into compress and grad_x '
    '(1.6756e-05 vs 1.8786e-06) with table gradients within 6%. '
    'CONFOUND, STATED: this stacks margin ON TOP OF z_norm and does NOT isolate it; if '
    'margin wins, margin-without-z_norm is the follow-up needed to deconfound. The reason '
    'to stack is that the two are plausibly complementary — z_norm pins |z| to std ~1, '
    'which is where margin still has healthy d/dm (~0.15-0.7) and where bounded_norm has '
    'already started saturating — but that is a hypothesis, not a claim. '
    'PRIOR: margin ran once as arm D at 4k on FAST, never on Light: 1.432430 vs baseline '
    '1.434572 (-0.002142), the only arm below baseline, though inside the 0.0096 4k seed '
    'sd. NOTE this run carries the per-layer ln1/ln2 logging, so layer 0 can be compared '
    'against exp_g_0190\'s healthy trajectory (final L0 ln2 norm 16.35017, plateau from '
    'step ~12000) rather than only at the endpoint. '
    'REFERENCES (corrected protocol, bs48 x 100, skip 12, 2,451,456 val tokens): '
    'exp_g_0190 1.203936 (the direct pair), exp_g_0189 1.207493, exp_n_0185 1.206222, '
    'exp_n_0184 1.201075, exp_n_0129 Fast gate-off 1.170961, vanilla dense exp_n_0135 '
    '1.165147 / exp_n_0176 1.161798 (vanilla seed spread 0.00335).')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model

    base = json.load(open(os.path.join(SRC, 'config.json')))
    cfg = copy.deepcopy(base)
    cfg['lut_confidence_form'] = 'margin'
    cfg['exp_name'] = RUN
    cfg['_arch_note'] = NOTE
    cfg['_sweep_tag'] = 'lookupffn-tier1-margin-on-light'

    drift = [k for k in set(cfg) | set(base)
             if k not in ('exp_name', 'lut_confidence_form', '_arch_note', '_sweep_tag')
             and cfg.get(k) != base.get(k)]
    if drift:
        print(f'*** STOP: unintended drift from exp_g_0190: {drift}')
        sys.exit(1)
    assert base['lut_confidence_form'] == 'bounded_norm'
    print('config diff vs exp_g_0190: lut_confidence_form bounded_norm -> margin '
          '(+ exp_name/_arch_note/_sweep_tag). No other field differs.')
    for k in ('lut_z_norm', 'lut_tables_no_decay', 'lut_inner_residual',
              'lut_tables_per_head', 'lut_n_anchor_pairs', 'random_seed',
              'n_steps', 'eval_every', 'lut_impl'):
        print(f'   {k:<22} {cfg.get(k, "absent")}')

    d = os.path.join(HERE, RUN)
    assert not os.path.exists(d), f'{d} exists -- never overwrite a prior run'
    os.makedirs(d)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    src_train, dst_train = os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py')
    shutil.copy(src_train, dst_train)
    assert open(src_train, 'rb').read() == open(dst_train, 'rb').read()
    trainer_src = open(dst_train).read()
    has_ln = 'ln2_norm_L' in trainer_src and 'ln_stats' in trainer_src
    print(f'\ntrain.py byte-identical to train_fixed.py: OK   carries ln logging: {has_ln}')

    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()

    # Build BOTH forms and compare, so "no new parameters" is measured, not assumed.
    counts = {}
    for form in ('bounded_norm', 'margin'):
        c = copy.deepcopy(cfg)
        c['lut_confidence_form'] = form
        torch.manual_seed(c['random_seed'])
        mm = build_model(c, vocab, device='cpu')
        counts[form] = sum(p.numel() for p in mm.parameters())
        if form == 'margin':
            m = mm
        else:
            del mm

    light = m.blocks[0].ffn.lut_light
    checks = {
        'total params': (counts['margin'], EXPECT),
        'params == exp_g_0190': (counts['margin'] - counts['bounded_norm'], 0),
        'confidence_form': (light.confidence_form, 'margin'),
        'fused kernel form id': (light._score_form_id, 2),
        'all layers on margin': ({b.ffn.lut_light.confidence_form for b in m.blocks},
                                 {'margin'}),
        'z_norm still on': ({b.ffn.z_norm is not None for b in m.blocks}, {True}),
        'inner_residual OFF': ({b.ffn.inner_residual for b in m.blocks}, {False}),
        'tables_per_head': (light.tables_per_head, 128),
        'trainer has ln logging': (has_ln, True),
    }
    ok = True
    print(f'\n{RUN}')
    for name, (got, want) in checks.items():
        good = got == want
        ok &= good
        print(f'   {name:<26}{str(got)[:26]:>28}   expected {str(want)[:16]:<18}'
              f'{"OK" if good else "*** MISMATCH ***"}')

    # Forward + backward on the REAL config (CPU), decompress driven to trained scale so
    # the zero-init does not make every gradient read exactly 0.
    with torch.no_grad():
        for b in m.blocks:
            b.ffn.decompress.weight.normal_(0, 2.3 / b.ffn.decompress.weight.numel() ** 0.5)
    loss = m(torch.randint(0, vocab, (2, 64))).float().mean()
    loss.backward()
    ffn = m.blocks[0].ffn
    gs = {'tables': light.tables.grad, 'compress': ffn.compress.weight.grad,
          'z_norm': ffn.z_norm.weight.grad}
    print(f'\n   smoke: loss {loss.item():.6g}   '
          + '  '.join(f'{k} grad {v.norm():.4e}' for k, v in gs.items()))
    for k, v in gs.items():
        alive = v is not None and torch.isfinite(v).all() and v.norm() > 0
        ok &= bool(alive)
        print(f'   {k} gradient finite and nonzero: {"YES" if alive else "*** NO ***"}')
    del m
    if not ok:
        print('\n*** STOP — not launching ***')
        sys.exit(1)
    print(f'\nwrote {d}/  — verified')


if __name__ == '__main__':
    main()
