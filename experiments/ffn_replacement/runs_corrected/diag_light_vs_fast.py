"""How much gradient signal does LightMultiHeadLUT actually get, relative to Fast?

Light's defining property: the routing address is `sign(d).detach()`, so x receives NO
gradient through the routing direction — only through the confidence score. Fast adds a
directional surrogate on top of that. The question this answers, before spending an hour of
GPU on arm B: is the missing directional gradient fatal or marginal?

The comparison is made at the CompressionMHL layer at the anchor sizing, with BOTH impls
carrying identical projections and table budgets (they differ only by Fast's 2 learnable
temperature scalars). Both are given the same input and the same upstream gradient.

IMPORTANT: `decompress` is zero-initialised by design, so at init the FFN output is exactly
0 and every gradient below it is 0 too — a naive probe measures nothing. So decompress is
given a realistic trained-scale value (the trained baseline checkpoint sits at norms
1.64-3.02) before the backward, exactly as diag_confidence_backward.py does.

    python diag_light_vs_fast.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT   # noqa: E402

KW = dict(input_dim=384, output_dim=384, inner_in_dim=32, inner_out_dim=48,
          nap=8, tph=256, n_heads=4, joint_head_compression=False,
          initial_weights_noise=1e-3, random_seed=1000)
DEC_NORM = 2.3          # the trained baseline's decompress scale


def probe(impl, **over):
    torch.manual_seed(0)
    kw = dict(KW)
    if impl == 'fast':
        kw.update(forward_mode='hard', learnable_temps=True)
    m = CompressionMultiHeadLUT(**kw, lut_impl=impl, **over)
    with torch.no_grad():
        m.decompress.weight.normal_(0, DEC_NORM / (384 * 192) ** 0.5)
    torch.manual_seed(1)
    x = (torch.randn(64, 384) * 0.6).requires_grad_(True)
    g = torch.randn(64, 384)
    y = m(x)
    y.backward(g)
    tbl = [p for n, p in m.named_parameters()
           if n.endswith('weights') or n.endswith('tables')][0]
    return dict(out=y.detach().abs().mean().item(),
                out_std=y.detach().std().item(),
                gx=x.grad.norm().item(),
                gtab=tbl.grad.norm().item(),
                gdec=m.decompress.weight.grad.norm().item(),
                gcom=m.compress.weight.grad.norm().item(),
                gtab_nz=(tbl.grad != 0).float().mean().item())


def row(name, r, ref=None):
    s = (f"   {name:<34}{r['out']:>10.5g}{r['gx']:>12.5g}{r['gtab']:>13.5g}"
         f"{r['gdec']:>12.5g}{r['gcom']:>12.5g}{r['gtab_nz']:>10.1%}")
    if ref is not None:
        s += (f"\n   {'':<34}{'':>10}{r['gx']/ref['gx']:>11.3f}x"
              f"{r['gtab']/ref['gtab']:>12.3f}x{r['gdec']/ref['gdec']:>11.3f}x"
              f"{r['gcom']/ref['gcom']:>11.3f}x")
    print(s)


def main():
    print('=' * 108)
    print('LightMultiHeadLUT vs FastMultiHeadLut at the anchor sizing, gate = bounded_norm')
    print('=' * 108)
    print(f"   {'':<34}{'|out|':>10}{'grad_x':>12}{'grad_tables':>13}"
          f"{'grad_dec':>12}{'grad_com':>12}{'tbl nz':>10}")

    fast_off = probe('fast')
    fast_bn = probe('fast', forward_confidence=True, confidence_form='bounded_norm')
    light_bn = probe('light', forward_confidence=True, confidence_form='bounded_norm')
    light_bd = probe('light', forward_confidence=True, confidence_form='bounded')
    light_mg = probe('light', forward_confidence=True, confidence_form='margin')

    row('fast, gate off (the baseline arm)', fast_off)
    row('fast + bounded_norm  (arm A\')', fast_bn, fast_off)
    row('light + bounded_norm (arm B)', light_bn, fast_off)
    row('light + bounded', light_bd, fast_off)
    row('light + margin', light_mg, fast_off)

    print('\n' + '=' * 108)
    print('THE QUESTION: is losing the directional surrogate fatal or marginal?')
    print('=' * 108)
    r_fast = fast_bn['gx']
    print(f"   grad_x, fast + bounded_norm   {fast_bn['gx']:.6g}")
    print(f"   grad_x, light + bounded_norm  {light_bn['gx']:.6g}"
          f"   ->  Light keeps {light_bn['gx'] / r_fast:.1%} of Fast's input gradient")
    print(f"   grad_x, fast gate OFF         {fast_off['gx']:.6g}"
          f"   (pure directional surrogate, no score path)")
    print('\n   Light\'s grad_x is ENTIRELY the score path: the address is detached, so the')
    print('   routing direction contributes nothing. Fast\'s is surrogate + score. The ratio')
    print('   above is therefore how much of the input-side learning signal survives.')
    print('\n   Table and decompress gradients do NOT depend on the surrogate at all (they')
    print('   flow through the gathered rows), so if those match, the ONLY thing arm B is')
    print('   missing is the ability to move x so as to change WHICH row is selected.')

    # How much does the score path alone move the compressed code, in relative terms?
    print('\n   compress-projection gradient (the code that the anchors read):')
    print(f"      fast gate off   {fast_off['gcom']:.6g}")
    print(f"      fast bnorm      {fast_bn['gcom']:.6g}")
    print(f"      light bnorm     {light_bn['gcom']:.6g}"
          f"   ->  {light_bn['gcom'] / fast_bn['gcom']:.1%} of Fast's")

    print('\n' + '=' * 108)
    print('DIRECTION, NOT SIZE: the trainer is AdamW, whose update is m_hat/(sqrt(v_hat)+eps)')
    print('— so a UNIFORM rescale of a gradient is almost entirely absorbed. A 16% gradient')
    print('norm is therefore NOT a 6x slowdown. What matters is whether the score path points')
    print('the same way as the full signal it replaces.')
    print('=' * 108)
    cos = torch.nn.functional.cosine_similarity
    for pname in ('compress.weight', 'compress.bias'):
        gf = grads('fast', pname, forward_confidence=True, confidence_form='bounded_norm')
        gl = grads('light', pname, forward_confidence=True, confidence_form='bounded_norm')
        g0 = grads('fast', pname)
        c_lf = cos(gl.reshape(1, -1), gf.reshape(1, -1)).item()
        c_l0 = cos(gl.reshape(1, -1), g0.reshape(1, -1)).item()
        c_f0 = cos(gf.reshape(1, -1), g0.reshape(1, -1)).item()
        print(f'   {pname:<16} cos(light_bn, fast_bn) {c_lf:+.4f}   '
              f'cos(light_bn, fast_gate_off) {c_l0:+.4f}   '
              f'cos(fast_bn, fast_gate_off) {c_f0:+.4f}')
    print('\n   cos near +1 -> Light\'s score path is a rescaled copy of Fast\'s signal, and')
    print('   AdamW will largely undo the rescale: the handicap is then MARGINAL.')
    print('   cos near 0  -> the directional surrogate carries information Light cannot see,')
    print('   and arm B is structurally handicapped, not merely quieter.')


def grads(impl, pname, **over):
    """The gradient of ONE named parameter, under a fixed input/upstream pair."""
    torch.manual_seed(0)
    kw = dict(KW)
    if impl == 'fast':
        kw.update(forward_mode='hard', learnable_temps=True)
    m = CompressionMultiHeadLUT(**kw, lut_impl=impl, **over)
    with torch.no_grad():
        m.decompress.weight.normal_(0, DEC_NORM / (384 * 192) ** 0.5)
    torch.manual_seed(1)
    x = (torch.randn(64, 384) * 0.6).requires_grad_(True)
    g = torch.randn(64, 384)
    m(x).backward(g)
    return dict(m.named_parameters())[pname].grad.detach().clone()


if __name__ == '__main__':
    main()
