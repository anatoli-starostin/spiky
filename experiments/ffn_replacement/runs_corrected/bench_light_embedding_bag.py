"""Can Light's forward be made as fast as Fast's WITHOUT giving up its pure-autograd backward?

Light's multi-head forward materialises the gathered rows as [B, H, T, d_out] and only then
multiplies by the score and sums over T. At the anchor sizing that is
6144 x 4 x 256 x 48 x 4B = 1.2 GiB of traffic per layer per call -- which is why its eval
forward measures ~9x slower than Fast's.

Fast avoids this with F.embedding_bag(mode='sum', per_sample_weights=score), which fuses the
gather and the sum-over-tables. The confidence score is EXACTLY a per-sample weight, and
embedding_bag is differentiable w.r.t. per_sample_weights -- so Light should be able to use
the same primitive and keep its defining property (detached integer address, gradient to x
only through the score).

This checks (a) that the two agree numerically, forward and backward, and (b) what it buys.
"""
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import _confidence_score

DEV = torch.device('cuda:0')
B, H, T, NAP, DIN, DOUT = 6144, 4, 256, 8, 32, 48
FORM = 'bounded_norm'


def bag_forward(m, x):
    """The embedding_bag formulation of LightMultiHeadLUT._forward_multi_head."""
    Bn = x.shape[0]
    Hh, Tt = m.n_heads, m.tables_per_head
    idx_a = m.anchor_a.reshape(1, Hh, Tt * NAP).expand(Bn, Hh, Tt * NAP)
    idx_b = m.anchor_b.reshape(1, Hh, Tt * NAP).expand(Bn, Hh, Tt * NAP)
    d = (torch.gather(x, 2, idx_a) - torch.gather(x, 2, idx_b)).view(Bn, Hh, Tt, NAP)
    index = ((d.detach() > 0).to(torch.int64)
             * m.powers.view(1, 1, 1, -1)).sum(dim=-1)                   # [B, H, T]
    flat = m.tables.reshape(Hh * Tt * m.table_size, m.output_dim)
    flat_idx = (index + m.table_offset.view(1, Hh, Tt)).reshape(-1)      # [B*H*T]
    score = _confidence_score(d, m.confidence_form, m.confidence_gain)   # [B, H, T]
    offsets = torch.arange(Bn * Hh, device=x.device, dtype=torch.long) * Tt
    out = F.embedding_bag(flat_idx, flat, offsets=offsets, mode='sum',
                          per_sample_weights=score.reshape(-1).to(flat.dtype))
    return out.view(Bn, Hh, m.output_dim)


def timed(fn, reps=25, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    torch.manual_seed(0)
    m = LightMultiHeadLUT(input_dim=DIN, n_tables=H * T, output_dim=DOUT, n_anchor_pairs=NAP,
                          confidence_form=FORM, random_seed=1000, device=DEV,
                          n_heads=H, multi_head_input=True).to(DEV)
    x = (torch.randn(B, H, DIN, device=DEV) * 0.6)
    g = torch.randn(B, H, DOUT, device=DEV)

    # --- correctness: forward ---
    with torch.no_grad():
        y_ref, y_bag = m(x), bag_forward(m, x)
    print(f'forward max |delta| : {(y_ref - y_bag).abs().max().item():.3e}   '
          f'(ref scale {y_ref.abs().mean().item():.4f})')

    # --- correctness: backward, both w.r.t. x and the tables ---
    def grads(fn):
        m.zero_grad(set_to_none=True)
        xx = x.detach().requires_grad_(True)
        fn(xx).backward(g)
        return xx.grad.clone(), m.tables.grad.clone()

    gx_ref, gt_ref = grads(lambda t: m(t))
    gx_bag, gt_bag = grads(lambda t: bag_forward(m, t))
    print(f'grad_x  max |delta| : {(gx_ref - gx_bag).abs().max().item():.3e}   '
          f'(ref norm {gx_ref.norm().item():.4f})')
    print(f'grad_W  max |delta| : {(gt_ref - gt_bag).abs().max().item():.3e}   '
          f'(ref norm {gt_ref.norm().item():.4f})')

    # --- speed ---
    def f_ref():
        with torch.no_grad():
            m(x)

    def f_bag():
        with torch.no_grad():
            bag_forward(m, x)

    def fb_ref():
        m.zero_grad(set_to_none=True)
        xx = x.detach().requires_grad_(True)
        m(xx).backward(g)

    def fb_bag():
        m.zero_grad(set_to_none=True)
        xx = x.detach().requires_grad_(True)
        bag_forward(m, xx).backward(g)

    torch.cuda.reset_peak_memory_stats()
    t_fr = timed(f_ref)
    p_ref_f = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    t_fb = timed(f_bag)
    p_bag_f = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    t_br = timed(fb_ref)
    p_ref_b = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    t_bb = timed(fb_bag)
    p_bag_b = torch.cuda.max_memory_allocated() / 2 ** 20

    print(f"\n{'':<26}{'fwd(eval)':>12}{'fwd+bwd':>11}{'peak fwd MiB':>15}"
          f"{'peak f+b MiB':>15}")
    print(f"{'Light as written':<26}{t_fr:>12.3f}{t_br:>11.3f}{p_ref_f:>15.1f}{p_ref_b:>15.1f}")
    print(f"{'Light via embedding_bag':<26}{t_fb:>12.3f}{t_bb:>11.3f}"
          f"{p_bag_f:>15.1f}{p_bag_b:>15.1f}")
    print(f"\n   forward  {t_fr / t_fb:.2f}x faster      fwd+bwd  {t_br / t_bb:.2f}x faster")


if __name__ == '__main__':
    main()
