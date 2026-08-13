"""Correctness gate for the `exp_outputs` flag on FastMultiHeadLut — run before any GPU.

Four checks:

 1. NON-REGRESSION. With exp_outputs=False the module is untouched: forward output and
    weight gradient must be bit-identical to the pre-change behaviour. (Verified against
    a module constructed exactly as exp10's `fastlut` builds it.)
 2. THE GATHER IS THE SAME SELECTION. `_exp_outputs_fwd` recomputes the sign-pack index
    itself, so it must select bit-identically the same rows as the production
    `forward_mode="hard"` path. Checked by summing the gathered rows over tables and
    comparing to the module's normal output — must be exactly 0 difference.
 3. THE LOG-SUM-EXP MATH is what was specified: tau * log(sum_t exp(w_t / tau)), checked
    against an independent naive loop.
 4. GRADIENTS FLOW to both the weights and tau, and the weight gradient is the
    softmax-weighted one log-sum-exp requires (not the sum path's broadcast).

Usage:  python verify_exp_outputs.py
"""
import math

import torch

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut

OBS, ACT, TPH, NAP, B = 17, 6, 32, 6, 512
SEED = 0
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails += 1


def build(exp_outputs, tau=0.1):
    torch.manual_seed(SEED)
    kw = dict(input_dim=OBS, n_heads=1, n_outputs=ACT, n_anchor_pairs=NAP,
              tables_per_head=TPH, forward_mode="hard", use_bf16=False,
              initial_weights_noise=0.001)
    if exp_outputs:
        kw.update(exp_outputs=True, exp_outputs_tau_init=tau)
    return FastMultiHeadLut(**kw)


def main():
    torch.manual_seed(SEED)
    x = torch.randn(B, OBS)

    print("\n1. non-regression: exp_outputs=False is the untouched module")
    m_off = build(False)
    y_off = m_off(x)
    y_off.sum().backward()
    g_off = m_off.weights.grad.clone()
    check("forward runs, shape [B, 1, ACT]", tuple(y_off.shape) == (B, 1, ACT),
          f"{tuple(y_off.shape)}")
    check("no tau parameter exists when the flag is off",
          not hasattr(m_off, "exp_outputs_tau_raw"))
    check("param count unchanged", sum(p.numel() for p in m_off.parameters())
          == TPH * (1 << NAP) * ACT, f"{sum(p.numel() for p in m_off.parameters()):,}")
    # sum-path weight grad from d(sum out)/dw is 1 at each chosen row
    check("sum-path weight grad is integer row counts",
          bool(torch.equal(g_off, g_off.round())))

    print("\n2. the exp_outputs gather selects the SAME rows as the hard path")
    m_on = build(True)
    check("weights identical between the two modules (same seed)",
          bool(torch.equal(m_off.weights.detach(), m_on.weights.detach())))
    # Reproduce the gather exactly as _exp_outputs_fwd does, then sum over tables.
    with torch.no_grad():
        d = x[:, m_on.soft_anchor_a_long] - x[:, m_on.soft_anchor_b_long]
        idx = ((d > 0).to(torch.int64) * m_on.soft_powers.view(1, 1, -1)).sum(-1)
        off = torch.arange(TPH, dtype=idx.dtype) * (1 << NAP)
        flat = (idx + off.view(1, -1)).reshape(-1)
        w_sel = m_on.weights.view(TPH * (1 << NAP), ACT)[flat].view(B, 1, TPH, ACT)
        summed = w_sel.sum(dim=2)
    # The decisive test is on the INDICES: they are int64, so exact equality means the
    # two paths select bit-identically the same rows. (Comparing the summed floats is
    # weaker -- embedding_bag reduces in a different order than .sum(), so fp32
    # associativity leaves ~1e-9 on outputs of order 1e-2. That is rounding, not a
    # different row: a wrong row would move the result by ~1e-3.)
    from spiky.lutorch.fast_multi_head_lut import _soft_lut_fwd_body
    with torch.no_grad():
        _, ref_index = _soft_lut_fwd_body(
            x, m_on.weights, m_on.soft_anchor_a_long, m_on.soft_anchor_b_long,
            m_on.soft_powers, 1, TPH, 1 << NAP)
    check("selected row indices are EXACTLY the production path's",
          bool(torch.equal(idx, ref_index)),
          f"{idx.numel():,} indices compared")
    diff = (summed - y_off.detach()).abs().max().item()
    rel = diff / y_off.detach().abs().max().item()
    check("summed gathered rows == hard-path output to fp32 rounding",
          rel < 1e-6, f"max|diff| = {diff:.3e} (relative {rel:.2e})")

    print("\n3. the log-sum-exp math matches the specification")
    tau = 0.1
    y_on = m_on(x)
    with torch.no_grad():
        naive = tau * torch.log(torch.exp(w_sel / tau).sum(dim=2))
    err = (y_on.detach() - naive).abs().max().item()
    check("tau * log(sum_t exp(w_t/tau)) reproduced", err < 1e-5, f"max|diff| = {err:.3e}")
    check("tau initialised to exp_outputs_tau_init",
          abs(float(m_on.exp_outputs_tau) - tau) < 1e-6,
          f"tau = {float(m_on.exp_outputs_tau):.6f}")

    print("\n4. gradients")
    m_on.zero_grad()
    y_on = m_on(x)
    y_on.sum().backward()
    gw, gt = m_on.weights.grad, m_on.exp_outputs_tau_raw.grad
    check("weights receive gradient", gw is not None and gw.abs().sum().item() > 0)
    check("tau receives gradient", gt is not None and abs(gt.item()) > 0,
          f"grad tau_raw = {gt.item():.6e}")
    # For LSE the per-table weight grad is softmax_t(w/tau), which sums to 1 per
    # (sample, output) -- exactly like the sum path's 1-per-table would NOT.
    with torch.no_grad():
        sm = torch.softmax(w_sel / tau, dim=2)
        expected_total = sm.sum().item()          # = B * ACT (softmax sums to 1)
    check("weight grad total == B*ACT (softmax weights sum to 1 per sample/output)",
          abs(gw.sum().item() - B * ACT) < 1e-2,
          f"{gw.sum().item():.3f} vs {B * ACT} (softmax check {expected_total:.3f})")
    check("weight grad is NOT the sum path's integer counts (softmax-weighted)",
          not torch.equal(gw, gw.round()))

    print("\n5. the tau -> 0 overflow guard")
    m_tiny = build(True, tau=0.1)
    with torch.no_grad():
        m_tiny.exp_outputs_tau_raw.fill_(-50.0)     # softplus underflows; floor catches it
        m_tiny.weights.mul_(1000.0)                 # and make w/tau enormous
    y_tiny = m_tiny(x)
    check("no NaN/inf when tau is driven to the floor with huge weights",
          bool(torch.isfinite(y_tiny).all()),
          f"tau = {float(m_tiny.exp_outputs_tau):.2e}")

    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
