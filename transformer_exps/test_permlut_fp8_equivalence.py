"""
Test whether PermutationalLut STE output rankings are preserved under low-precision
weight quantization. Tests fp32 (reference) vs bf16, fp16, fp8_e4m3.

The claim: STE forward only cares about sign(raw), so any monotonic quantization
that preserves per-entry signs should give identical output permutations.
"""
import sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spiky.lutorch.permutational_lut import PermutationalLut

DEVICE = 'cuda:0'


def kendall_tau_row(x, y):
    """Average Kendall tau across rows of x, y. Both [N, D]."""
    N, D = x.shape
    i, j = torch.triu_indices(D, D, offset=1, device=x.device)
    dx = x[:, i] - x[:, j]
    dy = y[:, i] - y[:, j]
    agree = torch.sign(dx) * torch.sign(dy)
    return agree.float().mean().item()


def top_rank_match(x, y):
    """Fraction of rows where argmax matches."""
    return (x.argmax(-1) == y.argmax(-1)).float().mean().item()


def perm_match_fraction(x, y):
    """Fraction of rows where the full ranking is identical."""
    return (x.argsort(-1) == y.argsort(-1)).all(-1).float().mean().item()


def make_lut(soft_mode='ste'):
    return PermutationalLut(
        n_inputs=32, n_outputs=32,
        input_nap=6, output_nap=32,
        n_heads=1, tph=2048,
        pair_mode='scrambled',
        soft_mode=soft_mode,
        temperature=0.1,
        random_seed=42, device=DEVICE,
        recompute_in_backward=True,
        initial_weights_noise=0.5,  # bigger init so weights actually vary
    )


def quantize_weights(lut, target_dtype):
    """Round-trip the inner LUT weights through target_dtype."""
    w = lut.inner.projection.weights
    with torch.no_grad():
        # Cast to target dtype, then back to fp32
        lut.inner.projection.weights.data = w.to(target_dtype).to(torch.float32)


def run(soft_mode):
    print(f'\n===== soft_mode={soft_mode} =====')

    x = torch.randn(64, 32, device=DEVICE)

    # Reference: fp32
    lut_ref = make_lut(soft_mode).to(DEVICE)
    lut_ref.eval()
    out_ref = lut_ref(x).squeeze(1)  # [B, N]
    ref_ranks = out_ref.argsort(-1)

    # Quantize copies
    for target_dtype, name in [
        (torch.bfloat16, 'bf16'),
        (torch.float16, 'fp16'),
        (torch.float8_e4m3fn, 'fp8_e4m3'),
        (torch.float8_e5m2, 'fp8_e5m2'),
    ]:
        lut_q = make_lut(soft_mode).to(DEVICE)
        # Sync weights first
        with torch.no_grad():
            lut_q.inner.projection.weights.data.copy_(
                lut_ref.inner.projection.weights.data
            )
        quantize_weights(lut_q, target_dtype)
        lut_q.eval()
        out_q = lut_q(x).squeeze(1)

        # Measure
        tau = kendall_tau_row(out_ref, out_q)
        top = top_rank_match(out_ref, out_q)
        full = perm_match_fraction(out_ref, out_q)
        max_err = (out_ref - out_q).abs().max().item()
        print(f'  {name:>9}: kendall_tau={tau:.4f}  top1_match={top:.2%}  '
              f'perm_match={full:.2%}  max_abs_diff={max_err:.4g}')


for sm in ['ste', 'rational', 'sigmoid']:
    run(sm)
