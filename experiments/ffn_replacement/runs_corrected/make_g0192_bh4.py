"""exp_g_0192: BH4 addressing -- LookupFFN's routing, our tables and decompress on top.

WHAT CHANGES vs exp_g_0190. Both compress AND the anchor-pair addressing are replaced. The
projection becomes BH4, a structured O(d log d) transform (four learnable block-diagonal
factors interleaved with fixed Walsh-Hadamard mixes), and the address becomes the sign of
the projected COORDINATES rather than the sign of coordinate DIFFERENCES. decompress is
unchanged and still sits on top, and the tables keep our narrow-rows-plus-decompress
layout rather than LookupFFN's full-width rows.

THE FOUR DESIGN DECISIONS, and why.

 1. WHERE THE BITS COME FROM. LookupFFN's own code projects hidden_size ->
    num_table*code_length in one rectangular BH4 and splits the result into per-table code
    vectors; coordinate-signing cannot reuse coordinates across tables the way anchor PAIRS
    can, so the budget is unavoidable. Here that is 4 heads x 128 tables x 7 bits = 3,584
    sign coordinates. We give each head its own SQUARE BH4 covering its own 128 x 7 = 896,
    rather than one rectangular transform over all 3,584: the parameter cost is identical,
    the padding waste is far smaller (384 -> 1024 instead of 384 -> 4096), and it matches
    the per-head structure the rest of the stack already has.

 2. POWER OF TWO. fwht requires it and n_embd is 384, so the working width is 1024, the
    first power of two at or above max(384, 896), and x is zero-padded into it.

 3. THE PADDING NEARLY KILLED IT, and the fix is a fixed pre-Hadamard. The normalised
    Hadamard is involutory, so with the reference's near-identity init the whole product
    collapses to R ~= H^4 = I: BH4 at init returns its input, padding included. Measured,
    that left ~512 of each head's 896 code coordinates sitting on padded zeros (pooled code
    std 0.65 ~= sqrt(384/896), which is the signature). One fixed H applied to the padded
    input BEFORE the learnable stack spreads the 384 informative coordinates across all
    1024; because it sits outside the product, H^4 = I can no longer restore the padding
    structure. Code std is now 0.606 ~= sqrt(384/1024) at every layer -- all coordinates
    carry signal. Zero parameters, and the reference's BH4 itself is untouched.

 4. THE OFFSET PROBLEM WAS REAL, AND IT WAS THE decompress BIAS. model_build zeroes the
    decompress WEIGHT but not its bias, so every block emits a token-independent constant
    (norm ~0.82 at init) that accumulates down the residual stream. LayerNorm removes each
    token's mean across dimensions, not a direction shared by all tokens. Anchor-pair
    addressing cancels such an offset inside d = z[a] - z[b] and never noticed it in three
    previous runs; coordinate-signing dies on it. Measured at init, by depth, the fraction
    of code coordinates whose sign never flips ran 0.00 / 0.14 / 0.28 / 0.38 / 0.43 / 0.49
    and layer 5 reached only 9.8 of its 128 addresses. Zeroing the decompress bias on this
    path -- which is what the residual-branch zero-init was supposed to achieve anyway, and
    what the dense baseline gets for free by using bias=False -- gives 0.0000 constant and
    127.4-127.8 of 128 addresses at EVERY layer.

PARAMETER ARITHMETIC. Parity target = compress (73,728 + 192 bias) + the anchor index
buffers (4 x 128 x 8 x 2 = 8,192) = 82,112 per layer. BH4 = n_heads x n_factors x dim x
block = 4 x 4 x 1024 x block, so block=4 gives 65,536 (0.798x target) and block=8 gives
131,072 (1.596x). block=4 is chosen: it is the closer of the two, and the 16,576 shortfall
is 0.026% of the model. (n_factors=5, block=4 would land at 81,920, within 192 of target,
but "BH4" means four factors and matching a 82k term to 0.2% inside a 48M model is not
worth changing the published transform for.)

    python make_g0192_bh4.py
"""
import copy
import hashlib
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
RUN = 'exp_g_0192_B16k_bh4_margin_tph128_nap7_seed1'
EXPECT = 48_427_008

NOTE = (
    'BH4 ADDRESSING: LookupFFN routing with our tables and decompress on top '
    '(lut_impl=bh4). Replaces BOTH compress and the anchor-pair addressing -- the '
    'projection becomes BH4 (four learnable block-diagonal factors interleaved with fixed '
    'Walsh-Hadamard mixes, O(d log d)) and the address becomes the sign of the projected '
    'COORDINATES instead of the sign of coordinate DIFFERENCES. decompress is unchanged; '
    'tables keep our narrow-rows-plus-decompress layout, NOT LookupFFN full-width rows. '
    'Score is margin, which is algebraically the paper\'s own score. 4 heads x 128 tables '
    'x 2^7 cells; each head has its own square BH4 at width 1024 (the first power of two '
    'at or above its 128 x 7 = 896 code coordinates), x zero-padded into it. '
    'TWO FIXES WERE REQUIRED AND BOTH WERE FOUND BY MEASUREMENT, not by inspection. '
    '(1) The normalised Hadamard is involutory, so with the reference near-identity init '
    'the product collapses to R ~= H^4 = I and BH4 returns its input, padding included: '
    '~512 of each head\'s 896 code coordinates sat on padded zeros (pooled code std 0.65 '
    '= sqrt(384/896)). A fixed pre-Hadamard on the padded input, outside the learnable '
    'product, fixes it -- code std is now 0.606 = sqrt(384/1024) at every layer. '
    '(2) model_build zeroes the decompress WEIGHT but not its BIAS, so every block emitted '
    'a token-independent constant (norm 0.82) that accumulated down the residual stream; '
    'anchor-pair addressing cancels such an offset inside d = z[a]-z[b] and never noticed '
    'it across three runs, but coordinate-signing dies on it. Constant-sign fraction by '
    'depth was 0.00/0.14/0.28/0.38/0.43/0.49 with layer 5 using only 9.8 of 128 addresses; '
    'zeroing the bias on this path gives 0.0000 constant and 127.4-127.8 of 128 addresses '
    'at every layer. '
    'PARAMETERS: parity target = compress (73,728+192) + anchor buffers (4*128*8*2 = 8,192) '
    '= 82,112/layer; BH4 = 4 heads * 4 factors * 1024 * block, so block=4 -> 65,536 '
    '(0.798x). TOTAL 48,427,008 vs exp_g_0190/0191\'s 67,352,256, i.e. -18,925,248 '
    '(-28.1%). THE DOMINANT TERM IS NOT BH4: it is NAP 8 -> 7, which halves every table '
    '(6,291,456 -> 3,145,728 per layer, -18,874,368 across the model). By the measured '
    'budget law (-0.007455 bpb per doubling of table parameters) that alone predicts a '
    'HANDICAP of about +0.0075 bpb, 2.2x the 0.00335 seed spread, before BH4 is judged at '
    'all. This run is therefore NOT parameter-comparable to the 0189/0190/0191 ladder, and '
    'the right control is a Light run at NAP=7 / tph=128 -- same table budget, same '
    'everything, only the addressing differing -- which does not yet exist. '
    'REFERENCES (corrected protocol, bs48 x 100, skip 12, 2,451,456 val tokens): exp_g_0190 '
    '1.203936, exp_g_0189 1.207493, exp_g_0191 (crashed at 15,100) 1.178680 @15,000, '
    'exp_n_0129 Fast gate-off 1.170961, vanilla dense 1.165147 / 1.161798.')


def fingerprint(m):
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        h.update(n.encode()); h.update(p.detach().numpy().tobytes())
    b = hashlib.sha256()
    for n, buf in sorted(m.named_buffers()):
        b.update(n.encode()); b.update(buf.detach().numpy().tobytes())
    s = hashlib.sha256('|'.join(f'{n}:{tuple(p.shape)}'
                                for n, p in sorted(m.named_parameters())).encode())
    return h.hexdigest(), b.hexdigest(), s.hexdigest()


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from model_build import build_model
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
    from spiky.lutorch.bh4_multi_head_lut import BH4MultiHeadLUT

    base = json.load(open(os.path.join(SRC, 'config.json')))
    vocab = RustBPETokenizer.from_directory(
        os.path.join(get_base_dir(), 'tokenizer')).get_vocab_size()

    # (a) backward compatibility: the UNMODIFIED base config must still build as before
    torch.manual_seed(base['random_seed'])
    m_base = build_model(base, vocab, device='cpu')
    ph, bh, sh = fingerprint(m_base)
    n_base = sum(p.numel() for p in m_base.parameters())
    print('BACKWARD COMPATIBILITY -- exp_g_0190\'s own config, rebuilt with the bh4 code present')
    print(f'   params {n_base:,}   (expected 67,352,256: {n_base == 67_352_256})')
    print(f'   param  sha256 {ph}')
    print(f'   buffer sha256 {bh}')
    print(f'   structure sha {sh}')
    print(f'   bh4 modules present: '
          f'{any(isinstance(mm, BH4MultiHeadLUT) for mm in m_base.modules())} (must be False)')
    del m_base

    cfg = copy.deepcopy(base)
    cfg['lut_impl'] = 'bh4'
    cfg['lut_n_anchor_pairs'] = 7
    cfg['lut_z_norm'] = False
    cfg['lut_confidence_form'] = 'margin'
    cfg['lut_bh4_block'] = 4
    cfg['lut_bh4_factors'] = 4
    cfg['exp_name'] = RUN
    cfg['_arch_note'] = NOTE
    cfg['_sweep_tag'] = 'lookupffn-bh4-addressing'
    changed = {'lut_impl', 'lut_n_anchor_pairs', 'lut_z_norm', 'lut_confidence_form',
               'lut_bh4_block', 'lut_bh4_factors', 'exp_name', '_arch_note', '_sweep_tag'}
    drift = [k for k in set(cfg) | set(base)
             if k not in changed and cfg.get(k) != base.get(k)]
    if drift:
        print(f'*** STOP: unintended drift from exp_g_0190: {drift}')
        sys.exit(1)
    print('\nconfig diff vs exp_g_0190: lut_impl light->bh4, nap 8->7, z_norm off, '
          'form bounded_norm->margin, + bh4 block/factors. No other field differs.')
    for k in ('lut_tables_no_decay', 'lut_tables_per_head', 'lut_n_heads', 'random_seed',
              'n_steps', 'eval_every', 'lut_inner_in_dim', 'lut_inner_out_dim'):
        print(f'   {k:<24} {cfg.get(k, "absent")}')

    d = os.path.join(HERE, RUN)
    assert not os.path.exists(d), f'{d} exists -- never overwrite a prior run'
    os.makedirs(d)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    src_train, dst_train = os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py')
    shutil.copy(src_train, dst_train)
    assert open(src_train, 'rb').read() == open(dst_train, 'rb').read()
    ts = open(dst_train).read()
    print(f'\ntrain.py byte-identical to train_fixed.py: OK   ln logging: '
          f'{"ln2_norm_L" in ts and "ln_stats" in ts}   '
          f'BH4 in the no-decay exemption: {"BH4MultiHeadLUT" in ts}')

    torch.manual_seed(cfg['random_seed'])
    m = build_model(cfg, vocab, device='cpu')
    tot = sum(p.numel() for p in m.parameters())
    ffn = m.blocks[0].ffn
    lb = ffn.lut_bh4

    names = {id(p): n for n, p in m.named_parameters()}
    exempt = (FastMultiHeadLut, LightMultiHeadLUT, BH4MultiHeadLUT)
    ids = {id(p) for mod in m.modules() if isinstance(mod, exempt)
           for p in mod.parameters(recurse=False)}
    dec, nod = [], []
    for p in m.parameters():
        (nod if (id(p) in ids or p.ndim < 2) else dec).append(names[id(p)])

    checks = {
        'total params': (tot, EXPECT),
        'compress removed': (isinstance(ffn.compress, torch.nn.Identity), True),
        'z_norm absent': (getattr(ffn, 'z_norm', None) is None, True),
        'decompress kept': (ffn.decompress.weight.shape, torch.Size([384, 192])),
        'decompress bias zeroed': (float(ffn.decompress.bias.abs().max()), 0.0),
        'BH4 in every block': ({isinstance(b.ffn.lut_bh4, BH4MultiHeadLUT)
                                for b in m.blocks}, {True}),
        'working width': (lb.width, 1024),
        'code coords used': (lb.n_code, 896),
        'cells per table': (lb.table_size, 128),
        'tables per head': (lb.tables_per_head, 128),
        'n_heads': (lb.n_heads, 4),
        'BH4 params/layer': (lb.bh4.blocks.numel(), 65_536),
        'confidence_form': (lb.confidence_form, 'margin'),
        'tables in NODECAY': (sum(n.endswith('lut_bh4.tables') for n in nod), 6),
        'bh4.blocks in DECAY': (sum('bh4.blocks' in n for n in dec), 6),
    }
    ok = True
    print(f'\n{RUN}')
    for name, (got, want) in checks.items():
        good = got == want
        ok &= good
        print(f'   {name:<26}{str(got)[:24]:>26}   expected {str(want)[:18]:<20}'
              f'{"OK" if good else "*** MISMATCH ***"}')

    with torch.no_grad():
        for b in m.blocks:
            b.ffn.decompress.weight.normal_(0, 2.3 / b.ffn.decompress.weight.numel() ** 0.5)
    loss = m(torch.randint(0, vocab, (2, 64))).float().mean()
    loss.backward()
    gs = {'bh4.blocks': lb.bh4.blocks.grad, 'tables': lb.tables.grad,
          'decompress': ffn.decompress.weight.grad}
    print('\n   smoke: ' + '  '.join(f'{k} grad {v.norm():.4e}' for k, v in gs.items()))
    for k, v in gs.items():
        alive = v is not None and torch.isfinite(v).all() and v.norm() > 0
        ok &= bool(alive)
        print(f'   {k} gradient finite and nonzero: {"YES" if alive else "*** NO ***"}')
    del m
    if not ok:
        print('\n*** STOP -- not launching ***')
        sys.exit(1)
    print(f'\nwrote {d}/  -- verified')


if __name__ == '__main__':
    main()
