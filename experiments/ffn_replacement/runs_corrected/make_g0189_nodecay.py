"""exp_g_0189: exp_n_0185 with the LUT tables exempted from weight decay. One variable.

THE BUG THIS CORRECTS. `train_fixed.py:setup_optimizer` exempts LUT table parameters from
weight decay, but the exemption matched by CLASS -- `isinstance(m, FastMultiHeadLut)` -- and
LightMultiHeadLUT's `tables` is 3-D, so it also escaped the `ndim < 2` rule. Result:

    Fast  (exp_n_0129)   75,497,472 table params at weight_decay = 0.0
    Light (exp_n_0184)   75,497,472 table params at weight_decay = 0.1
    Light (exp_n_0185)   37,748,736 table params at weight_decay = 0.1

Every Light-vs-Fast number we have has therefore been measuring the implementation PLUS a
regularisation difference on the single largest parameter block. It was unintended: the
exemption predates LightMultiHeadLUT and adding Light silently opted its tables in.

The fix is opt-in via `lut_tables_no_decay` (default False = the old behaviour, verified
bit-identical in group membership) rather than silent, so every previous run stays
reproducible under the same trainer.

This run is exp_n_0185 (1.206222) with `lut_tables_no_decay: true` as the only functional
change beyond eval cadence, so it isolates exactly that regularisation difference. The decay
group is an optimiser concern, so the model is untouched: the parameter count must remain
exactly 67,351,680.

NOTE ON WHAT IT DOES NOT TEST: this is not expected to explain the layer-0 LayerNorm collapse
(ln2.weight is 1-D and was never decayed, and decay on `tables` cannot reach it). It is a
correction to the comparison, not a fix for that pathology.

    python make_g0189_nodecay.py
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

SRC = os.path.join(HERE, 'exp_n_0185_B16k_light_bnorm_tph128_seed1')
RUN = 'exp_g_0189_B16k_light_bnorm_tph128_nodecay_seed1'
EXPECT = 67_351_680

NOTE = (
    'ARM B AT 16K WITH THE LUT TABLES EXEMPTED FROM WEIGHT DECAY '
    '(lut_tables_no_decay=true) — LightMultiHeadLUT with the normalised confidence gate '
    '(lut_impl="light", lut_forward_confidence=true, lut_confidence_form="bounded_norm"), '
    'copied from exp_n_0185 with lut_tables_no_decay as the only functional change beyond '
    'eval cadence (asserted; any other drift aborts the build). THE BUG THIS CORRECTS: '
    'train_fixed.py\'s setup_optimizer exempts LUT tables from weight decay, but matched by '
    'class — isinstance(m, FastMultiHeadLut) — and LightMultiHeadLUT\'s `tables` is 3-D so it '
    'also escaped the ndim<2 rule. Fast therefore trained its 75,497,472 table parameters at '
    'weight_decay=0.0 while Light trained the identical tables at 0.1 (37,748,736 at this '
    'tph=128 sizing). Every Light-vs-Fast comparison so far has measured the implementation '
    'PLUS that regularisation difference on the largest parameter block; it was unintended, '
    'since the exemption predates LightMultiHeadLUT. The fix is opt-in rather than silent so '
    'previous runs stay reproducible: with the flag absent or false the decay/nodecay group '
    'membership is bit-identical to before (verified by hashing the grouped parameter-name '
    'sets for exp_n_0185: decay sha 45714bc3..., nodecay sha 55ec322c..., 67,338,240 / 13,440 '
    'params, unchanged). With the flag true, lut_light.tables moves to the nodecay group and '
    'the decayed total falls from 67,338,240 to 29,589,504 — matching Fast\'s grouping '
    'exactly. The decay group is an optimiser concern, so the MODEL is untouched: parameter '
    'count stays exactly 67,351,680. eval_every is 500 rather than 0185\'s 200, which shares '
    'eval points with the vanilla reference exp_n_0176 (also 500) at every multiple of 500; '
    'cadence was verified trajectory-neutral (eval draws no RNG and uses a separate val '
    'loader, so the training stream is bit-identical regardless of how often it runs). WHAT '
    'THIS DOES NOT TEST: it is not expected to explain the layer-0 ln2.weight collapse — '
    'LayerNorm gains are 1-D and were never decayed, and decay on `tables` cannot reach them. '
    'This corrects the comparison, it does not fix that pathology. inner_residual is OFF here '
    'so the two changes are not confounded; exp_n_0188 (inner residual, no decay fix) was '
    'abandoned at step 9,000 in favour of this. REFERENCES (corrected protocol, bs48 x 100, '
    'skip 12, 2,451,456 val tokens): exp_n_0185 1.206222 (the direct pair), exp_n_0184 '
    'tph256 1.201075, exp_n_0186 tph128/nap7 1.208987, exp_n_0129 Fast gate-off 1.170961, '
    'vanilla dense exp_n_0135 1.165147 / exp_n_0176 1.161798 (vanilla seed spread 0.00335).')


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

    base = json.load(open(os.path.join(SRC, 'config.json')))
    cfg = copy.deepcopy(base)
    cfg['lut_tables_no_decay'] = True
    cfg['eval_every'] = 500
    cfg['exp_name'] = RUN
    cfg['_arch_note'] = NOTE
    cfg['_sweep_tag'] = 'lookupffn-arm-b-16k-nodecay'

    drift = [k for k in set(cfg) | set(base)
             if k not in ('exp_name', 'lut_tables_no_decay', 'eval_every',
                          '_arch_note', '_sweep_tag')
             and cfg.get(k) != base.get(k)]
    if drift:
        print(f'*** STOP: unintended drift from exp_n_0185: {drift}')
        sys.exit(1)
    assert cfg.get('lut_inner_residual', False) is False, 'inner_residual must be OFF'
    print('config diff vs exp_n_0185: + lut_tables_no_decay=True, eval_every 200 -> 500 '
          '(+ exp_name/_arch_note/_sweep_tag). No other field differs.')
    print(f'lut_inner_residual: {cfg.get("lut_inner_residual", "absent")} (OFF, as required)')

    d = os.path.join(HERE, RUN)
    assert not os.path.exists(d), f'{d} exists -- never overwrite a prior run'
    os.makedirs(d)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    src_train, dst_train = os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py')
    shutil.copy(src_train, dst_train)
    assert open(src_train, 'rb').read() == open(dst_train, 'rb').read()
    print('train.py byte-identical to the UPDATED train_fixed.py: OK')

    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()
    torch.manual_seed(cfg['random_seed'])
    m = build_model(cfg, vocab, device='cpu')
    tot = sum(p.numel() for p in m.parameters())

    def split(flag):
        exempt = (FastMultiHeadLut, LightMultiHeadLUT) if flag else (FastMultiHeadLut,)
        ids = {id(p) for mod in m.modules() if isinstance(mod, exempt)
               for p in mod.parameters(recurse=False)}
        names = {id(p): n for n, p in m.named_parameters()}
        dec, nod = [], []
        for p in m.parameters():
            (nod if (id(p) in ids or p.ndim < 2) else dec).append(names[id(p)])
        nd = dict(m.named_parameters())
        return dec, nod, sum(nd[n].numel() for n in dec), sum(nd[n].numel() for n in nod)

    d0, n0, p0, q0 = split(False)
    d1, n1, p1, q1 = split(True)
    tbl = [n for n in d0 if n.endswith('tables')]
    print(f'\nTHE FLAG WORKS:')
    print(f'   flag False: decayed {p0:,}  nodecay {q0:,}   '
          f'lut_light.tables in DECAY ({len(tbl)} tensors)')
    print(f'   flag True : decayed {p1:,}  nodecay {q1:,}   '
          f'lut_light.tables in NODECAY ({len([n for n in n1 if n.endswith("tables")])})')
    print(f'   -> {p0-p1:,} table parameters moved out of the decay group '
          f'(Fast\'s decayed total is 29,589,504)')

    ffn = m.blocks[0].ffn
    light = ffn.lut_light
    checks = {
        'total params (model untouched)': (tot, EXPECT),
        'tables in NODECAY under the flag': (len([n for n in n1 if n.endswith('tables')]), 6),
        'tables in DECAY under the flag': (len([n for n in d1 if n.endswith('tables')]), 0),
        'inner_residual OFF': ({b.ffn.inner_residual for b in m.blocks}, {False}),
        'lut_light present': (light is not None, True),
        'lut_batched absent': (not hasattr(ffn, 'lut_batched'), True),
        'multi_head_input ON': (light.multi_head_input, True),
        'tables_per_head': (light.tables_per_head, 128),
        'n_anchor_pairs': (light.n_anchor_pairs, 8),
        'confidence_form': (light.confidence_form, 'bounded_norm'),
        'tables dtype fp32': (light.tables.dtype, torch.float32),
    }
    ok = True
    print(f'\n{RUN}')
    for name, (got, want) in checks.items():
        good = got == want
        ok &= good
        print(f'   {name:<36}{str(got)[:22]:>24}   expected {str(want)[:18]:<20}'
              f'{"OK" if good else "*** MISMATCH ***"}')

    with torch.no_grad():
        for b in m.blocks:
            b.ffn.decompress.weight.normal_(0, 2.3 / b.ffn.decompress.weight.numel() ** 0.5)
    loss = m(torch.randint(0, vocab, (2, 64))).float().mean()
    loss.backward()
    gt, gc = light.tables.grad, ffn.compress.weight.grad
    print(f'\n   smoke (decompress driven to trained scale first): loss {loss.item():.6g}  '
          f'tables grad {gt.norm():.6g}  compress grad {gc.norm():.6g}  '
          f'finite={torch.isfinite(gt).all().item()}')
    ok &= bool(torch.isfinite(gt).all() and gt.norm() > 0)
    del m
    if not ok:
        print('\n*** STOP — not launching ***')
        sys.exit(1)
    print(f'\nwrote {d}/  — verified')


if __name__ == '__main__':
    main()
