"""Backward-magnitude sanity for the forward-confidence gate, at anchor sizing, on CPU."""
import sys, torch
sys.path.insert(0, '/home/astarostin/projects/spiky/src')
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

KW = dict(input_dim=384, output_dim=384, inner_in_dim=32, inner_out_dim=48,
          nap=8, tph=256, n_heads=4, joint_head_compression=False,
          forward_mode='hard', initial_weights_noise=1e-3, learnable_temps=True,
          random_seed=1000)
torch.manual_seed(0)
x = torch.randn(64, 384) * 0.6          # ~ the scale of a post-LayerNorm activation

def probe(**over):
    torch.manual_seed(0)
    m = CompressionMultiHeadLUT(**KW, **over)
    # decompress is zeroed by design; give it a realistic trained-scale value so the
    # gradient actually flows (norm ~2.3 per the trained baseline checkpoint)
    with torch.no_grad():
        m.decompress.weight.normal_(0, 2.3 / (384 * 192) ** 0.5)
    xx = x.clone().requires_grad_(True)
    y = m(xx)
    y.pow(2).mean().backward()
    tbl = [p for n, p in m.named_parameters() if n.endswith('weights')][0]
    return dict(out=y.abs().mean().item(), gx=xx.grad.norm().item(),
                gtab=tbl.grad.norm().item(),
                gdec=m.decompress.weight.grad.norm().item(),
                gcom=m.compress.weight.grad.norm().item())

off = probe()
bd = probe(forward_confidence=True)
mg = probe(forward_confidence=True, confidence_form='margin')

print(f"{'':<26}{'|out|':>12}{'grad_x':>12}{'grad_tables':>14}{'grad_dec':>12}{'grad_com':>12}")
for name, r in (('gate off (baseline)', off), ('gate bounded', bd), ('gate margin', mg)):
    print(f"{name:<26}{r['out']:>12.6g}{r['gx']:>12.6g}{r['gtab']:>14.6g}"
          f"{r['gdec']:>12.6g}{r['gcom']:>12.6g}")

print('\nratios vs gate-off:')
for name, r in (('bounded', bd), ('margin', mg)):
    print(f"   {name:<10} out {r['out']/off['out']:8.4f}x   grad_x {r['gx']/off['gx']:8.4f}x   "
          f"grad_tables {r['gtab']/off['gtab']:8.4f}x   "
          f"grad_dec {r['gdec']/off['gdec']:8.4f}x   grad_com {r['gcom']/off['gcom']:8.4f}x")

# how much of grad_x is the NEW score path vs the pre-existing directional surrogate?
print('\nsplit of grad_x under the gate (score path vs directional surrogate):')
print('   measured by differencing: grad_x(gate) - grad_x(off) is not the score path in')
print('   general (the surrogate is itself scaled by the score), so instead report the')
print('   norms directly -- the point is whether the gate changes the scale by orders.')

# train vs eval consistency under the gate
torch.manual_seed(0)
m = CompressionMultiHeadLUT(**KW, forward_confidence=True)
m.train(); a = m(x)
m.eval();  b = m(x)
print(f"\ntrain vs eval under the gate: identical = {torch.equal(a, b)}   "
      f"max|delta| = {(a-b).abs().max():.3g}")
