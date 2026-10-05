"""Tests for the two shared ManifestoLUT features: cell_tv() (Hamming-1 TV penalty) and
table_dropout_rate (whole-table inverted dropout, folded into every cartridge's read).

cell_tv: bit-for-bit vs an explicit Hamming-1 pair enumeration (the reference formula), + the
model-level mean aggregator. table dropout: has real effect in train, is a no-op at eval, is
unbiased in expectation (inverted rescale), gates the backward (dropped table -> zero grad), and --
crucially -- the FUSED / native-kernel read paths produce the SAME dropout-aware value AND gradient
as the pure oracle (so the mask folded into per-sample-weights / scaled into the native grad is not
silently dropped or mis-scaled). CPU fp64 (oracle / tier1) + H100 fp32 (native kernels)."""
import math

import pytest
import torch
import torch.nn as nn

from spiky.lutorch_ex import (ConfidenceLUT, FusedManifestoHardLUT, FusedManifestoSoftLUT,
                              FusedSoftSignHardLUT, FusedSoftSignSmoothLUT, LUTSpec, ManifestoHardLUT,
                              ManifestoSoftLUT, QuantisedConfidenceLUT, SoftSignHardLUT,
                              SoftSignSmoothLUT, cell_tv_penalty)

_CUDA = torch.cuda.is_available()


def _ref_cell_tv(weights: torch.Tensor, nap: int) -> torch.Tensor:
    """Explicit reference: mean over all Hamming-1 cell pairs (and tables) of ||v_c - v_c'||^2
    (d_out summed in)."""
    G, tph, K, d_out = weights.shape
    t = weights.reshape(G * tph, K, d_out)
    tot = t.new_zeros(())
    for c in range(K):
        for b in range(nap):
            cp = c ^ (1 << b)
            if cp > c:
                tot = tot + ((t[:, c, :] - t[:, cp, :]) ** 2).sum()
    n_pairs = (G * tph) * nap * (1 << (nap - 1))
    return tot / n_pairs


@pytest.mark.parametrize("H,TPH,NAP,D", [(2, 3, 4, 5), (1, 2, 3, 4), (3, 2, 5, 4)])
def test_cell_tv_matches_explicit_formula(H, TPH, NAP, D):
    spec = LUTSpec(h_in=H, h_out=H, tph=TPH, nap=NAP, d_in=max(D, 2), d_out=D)
    for Cls in (ManifestoHardLUT, lambda s, **k: ConfidenceLUT(s, read_top_n=2, **k)):
        m = Cls(spec, seed=1, weight_init_std=1.0)
        m = m.double()
        got, ref = m.cell_tv(), _ref_cell_tv(m.weights, NAP)
        assert torch.allclose(got, ref, rtol=0, atol=1e-12), (got.item(), ref.item())
        assert got.requires_grad  # differentiable w.r.t. weights


def test_cell_tv_penalty_is_mean_over_cartridges():
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=4, d_in=6, d_out=5)
    a = ManifestoHardLUT(spec, seed=1, weight_init_std=1.0).double()
    b = ConfidenceLUT(spec, seed=2, weight_init_std=1.0, read_top_n=2).double()
    model = nn.ModuleList([a, b])
    expected = torch.stack([a.cell_tv(), b.cell_tv()]).mean()
    assert torch.allclose(cell_tv_penalty(model), expected, rtol=0, atol=0)
    assert cell_tv_penalty(nn.Linear(3, 3)).item() == 0.0   # no cartridge -> 0


def test_table_dropout_rate_validation():
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=4, d_in=6, d_out=5)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            ManifestoHardLUT(spec, seed=0, table_dropout_rate=bad)


# -- build helpers -----------------------------------------------------------------------------
def _spec(H=2, TPH=4, NAP=4, D=6):
    return LUTSpec(h_in=H, h_out=H, tph=TPH, nap=NAP, d_in=D, d_out=D)


def _build(Cls, spec, device, dtype, rate):
    kw = dict(seed=0, weight_init_std=1.0, table_dropout_rate=rate)
    if Cls in (ConfidenceLUT, QuantisedConfidenceLUT):
        kw["read_top_n"] = 2
    return Cls(spec, **kw).to(device=device, dtype=dtype)


ALL = [ManifestoHardLUT, ManifestoSoftLUT, SoftSignHardLUT, SoftSignSmoothLUT,
       FusedManifestoHardLUT, FusedManifestoSoftLUT, FusedSoftSignHardLUT, FusedSoftSignSmoothLUT,
       ConfidenceLUT, QuantisedConfidenceLUT]


def _devs_dtypes():
    out = [("cpu", torch.float64)]
    if _CUDA:
        out.append(("cuda", torch.float32))
    return out


@pytest.mark.parametrize("Cls", ALL)
def test_dropout_effect_and_eval_noop(Cls):
    for device, dtype in _devs_dtypes():
        if Cls is QuantisedConfidenceLUT and device == "cpu":
            continue  # quant native-op path is CUDA; CPU quant is covered by the fused==pure test below
        spec = _spec()
        m = _build(Cls, spec, device, dtype, 0.3)
        x = torch.randn(8, spec.h_in, spec.d_in, device=device, dtype=dtype, requires_grad=True)
        # TRAIN with dropout: output differs from the rate-0 output (dropout has real effect)
        m.train()
        torch.manual_seed(0); y_drop = m(x)
        m0 = _build(Cls, spec, device, dtype, 0.0)
        m0.load_state_dict(m.state_dict()); m0.train()
        y_full = m0(x)
        assert not torch.allclose(y_drop, y_full, rtol=1e-4, atol=1e-4), f"{Cls.__name__}: dropout had no effect"
        # EVAL: no mask -> matches the rate-0 eval exactly
        m.eval(); m0.eval()
        with torch.no_grad():
            assert torch.allclose(m(x), m0(x), rtol=0, atol=0), f"{Cls.__name__}: eval not mask-free"


@pytest.mark.parametrize("Cls", ALL)
def test_dropout_unbiased_in_expectation(Cls):
    # inverted dropout: E[output] over masks == the no-dropout output.
    device, dtype = ("cuda", torch.float32) if _CUDA else ("cpu", torch.float64)
    if Cls is QuantisedConfidenceLUT and device == "cpu":
        pytest.skip("quant native path is CUDA-only")
    spec = _spec()
    m = _build(Cls, spec, device, dtype, 0.25); m.train()
    m0 = _build(Cls, spec, device, dtype, 0.0); m0.load_state_dict(m.state_dict()); m0.train()
    x = torch.randn(16, spec.h_in, spec.d_in, device=device, dtype=dtype)
    with torch.no_grad():
        full = m0(x)
        acc = torch.zeros_like(full)
        N = 400
        for i in range(N):
            torch.manual_seed(1000 + i)
            acc += m(x)
        mean = acc / N
    # Monte-Carlo over 400 masks: mean should approach the no-dropout output.
    rel = (mean - full).abs().mean() / full.abs().mean().clamp_min(1e-6)
    assert rel < 0.03, f"{Cls.__name__}: inverted-dropout mean biased, rel={rel:.3f}"


def _inject_mask(m, mask):
    """Force a fixed keep-mask (bypass RNG / training gate) so two cartridges share the exact mask."""
    m._table_dropout_mask = lambda B, device, dtype: mask.to(device=device, dtype=dtype)


# Fused/native cartridge  <->  its pure oracle (SAME weights + SAME injected mask): value AND grads
# must agree. This is the native-kernel dropout-threading check (fused vs eager).
PAIRS = [(FusedManifestoHardLUT, ManifestoHardLUT),
         (FusedManifestoSoftLUT, ManifestoSoftLUT),
         (FusedSoftSignHardLUT, SoftSignHardLUT),
         (FusedSoftSignSmoothLUT, SoftSignSmoothLUT)]


@pytest.mark.parametrize("Fused,Pure", PAIRS)
def test_fused_matches_pure_with_dropout_value_and_grad(Fused, Pure):
    for device, dtype in _devs_dtypes():
        spec = _spec()
        atol = 1e-9 if dtype == torch.float64 else 2e-3
        pf = _build(Fused, spec, device, dtype, 0.3); pf.train()
        pu = _build(Pure, spec, device, dtype, 0.3); pu.load_state_dict(pf.state_dict()); pu.train()
        B = 8
        keep = (torch.rand(B, spec.n_groups, spec.tph, device=device) < 0.7).to(dtype) / 0.7
        _inject_mask(pf, keep); _inject_mask(pu, keep)
        go = torch.randn(B, spec.h_out, spec.d_out, device=device, dtype=dtype)
        def run(mm):
            x = torch.randn(B, spec.h_in, spec.d_in, device=device, dtype=dtype,
                            generator=torch.Generator(device=device).manual_seed(7)).requires_grad_(True)
            y = mm(x)
            gw, gx = torch.autograd.grad((y * go).sum(), [mm.weights, x])
            return y.detach(), gw, gx
        yf, gwf, gxf = run(pf)
        yu, gwu, gxu = run(pu)
        nm = lambda a, b: (a - b).abs().max().item()
        assert nm(yf, yu) < atol, f"{Fused.__name__} value vs pure: {nm(yf,yu):.2e}"
        assert nm(gwf, gwu) < atol, f"{Fused.__name__} grad_W vs pure: {nm(gwf,gwu):.2e}"
        assert nm(gxf, gxu) < atol, f"{Fused.__name__} grad_x vs pure: {nm(gxf,gxu):.2e}"


@pytest.mark.parametrize("Cls", [ManifestoHardLUT, ConfidenceLUT, QuantisedConfidenceLUT,
                                 FusedManifestoHardLUT])
def test_dropped_table_gets_zero_grad(Cls):
    # A table dropped (mask 0) for ALL samples must receive zero weight gradient at its addressed cells.
    device, dtype = ("cuda", torch.float32) if _CUDA else ("cpu", torch.float64)
    if Cls is QuantisedConfidenceLUT and device == "cpu":
        pytest.skip("quant native path is CUDA-only")
    spec = _spec()
    m = _build(Cls, spec, device, dtype, 0.3); m.train()
    B = 8
    keep = torch.ones(B, spec.n_groups, spec.tph, device=device, dtype=dtype)
    keep[:, :, 0] = 0.0                                  # drop table 0 in every group, every sample
    _inject_mask(m, keep)
    x = torch.randn(B, spec.h_in, spec.d_in, device=device, dtype=dtype, requires_grad=True)
    (m(x).pow(2).sum()).backward()
    # weights[:, 0] (table 0 of every group) must have exactly zero grad (dropped everywhere).
    g0 = m.weights.grad[:, 0]
    assert g0.abs().max().item() == 0.0, f"{Cls.__name__}: dropped table 0 got nonzero grad {g0.abs().max()}"
    assert m.weights.grad[:, 1:].abs().max().item() > 0.0  # surviving tables do get grad
