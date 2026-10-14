"""The fused kernels' CTA order (cartridges/_cuda_ext._ARCH_DEFAULTS): the forward order comes from the per-arch table,
overridable by LUTORCH_EX_<FAMILY>_CUDA_CTA_ORDER; the backward is always group-major.

Resolution is pure Python and runs on CPU (the device capability is stubbed). On a CUDA device, every fused forward is
run in both orders: the order only maps CTAs to (group, row block) -- each output element is still computed by one
CTA with the same arithmetic -- so the outputs must be bit-identical."""
import re
from pathlib import Path

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges import _cuda_ext
from spiky.lutorch_ex.cartridges._cuda_ext import resolve_cta_order
from spiky.lutorch_ex.tests._requires_extension import requires_extension

CSRC = Path(__file__).resolve().parents[1] / "cartridges" / "csrc"
FAMILIES = ("MANI", "CONF", "QUANT")


# ---- the table, pure ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cap,order", [((9, 0), "group"), ((12, 0), "interleaved")])
def test_listed_architectures_resolve_to_their_table_entry(cap, order):
    assert resolve_cta_order(cap) == order


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 1), (10, 0), (12, 1), (13, 0)])
def test_an_unlisted_architecture_falls_back_to_interleaved(cap):
    """The documented fallback is the order every kernel used before the table: an unlisted card never changes."""
    assert resolve_cta_order(cap) == "interleaved" == _cuda_ext._ARCH_FALLBACK["cta_order"]


@pytest.mark.parametrize("cap", [(9, 0), (12, 0), (7, 5)])
@pytest.mark.parametrize("env", ["group", "interleaved"])
def test_the_env_override_wins_over_the_table(cap, env):
    assert resolve_cta_order(cap, env) == env


def test_an_unknown_order_raises():
    with pytest.raises(ValueError, match="'diagonal'"):
        resolve_cta_order((9, 0), "diagonal")


# ---- the per-(family, device) resolution, with the env var and a stubbed capability ---------------------------------

@pytest.fixture
def stub_capability(monkeypatch):
    """Stub the device capability and clear the per-process cache around the test."""
    def stub(cap):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cap)
        _cuda_ext._forward_group_major.cache_clear()
    yield stub
    _cuda_ext._forward_group_major.cache_clear()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cap,expected", [((9, 0), True), ((12, 0), False), ((11, 0), False)])
def test_forward_group_major_follows_the_table(stub_capability, monkeypatch, family, cap, expected):
    monkeypatch.delenv(f"LUTORCH_EX_{family}_CUDA_CTA_ORDER", raising=False)
    stub_capability(cap)
    assert _cuda_ext.forward_group_major(family, torch.device("cuda", 0)) is expected


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_group_major_env_override_per_family(stub_capability, monkeypatch, family):
    stub_capability((9, 0))
    monkeypatch.setenv(f"LUTORCH_EX_{family}_CUDA_CTA_ORDER", "interleaved")
    assert _cuda_ext.forward_group_major(family, torch.device("cuda", 0)) is False
    other = next(f for f in FAMILIES if f != family)
    monkeypatch.delenv(f"LUTORCH_EX_{other}_CUDA_CTA_ORDER", raising=False)
    assert _cuda_ext.forward_group_major(other, torch.device("cuda", 0)) is True     # another family is unaffected
    stub_capability((12, 0))
    monkeypatch.setenv(f"LUTORCH_EX_{family}_CUDA_CTA_ORDER", "group")
    assert _cuda_ext.forward_group_major(family, torch.device("cuda", 0)) is True


def test_a_bad_env_value_raises_naming_the_variable(stub_capability, monkeypatch):
    stub_capability((9, 0))
    monkeypatch.setenv("LUTORCH_EX_CONF_CUDA_CTA_ORDER", "1")
    with pytest.raises(ValueError, match="LUTORCH_EX_CONF_CUDA_CTA_ORDER"):
        _cuda_ext.forward_group_major("CONF", torch.device("cuda", 0))


# ---- the kernels: forward sites read the order, backward sites are group-major unconditionally (source check) -------

_SITE = re.compile(r"const int g = static_cast<int>\((.*?)\);")
_OWNER = re.compile(r"(\w+_(?:fwd|bwd)_(?:body|kernel)\w*)\s*\(")


def test_forward_sites_take_the_order_and_backward_sites_are_group_major():
    seen = {"fwd": 0, "bwd": 0}
    for src in sorted(CSRC.glob("*.cu")):
        s = src.read_text()
        for m in _SITE.finditer(s):
            owner = _OWNER.findall(s[:m.start()])[-1]
            if "_fwd_" in owner:
                seen["fwd"] += 1
                assert m.group(1) == "p.group_major ? cta / nb : cta % p.G", f"{src.name}: {owner}"
            else:
                seen["bwd"] += 1
                assert m.group(1) == "cta / nb", f"{src.name}: {owner} must be group-major unconditionally"
    # manifesto, confidence, quant, deploy forwards; manifesto, confidence, quant backwards
    assert seen == {"fwd": 4, "bwd": 3}, seen


# ---- the kernels: both forward orders give bit-identical outputs ----------------------------------------------------

def _cases():
    spec = dict(h_in=4, h_out=4, tph=16, nap=6, d_in=24, d_out=24)
    return [
        ("MANI", "manifesto", lambda s: lx.FusedManifestoHardLUT(s, seed=3)),
        ("MANI", "manifesto", lambda s: lx.FusedManifestoSoftLUT(s, seed=3)),
        ("CONF", "confidence", lambda s: lx.FusedConfidenceLUT(s, seed=3, read_top_n=1)),
        ("CONF", "confidence", lambda s: lx.FusedConfidenceLUT(s, seed=3, read_top_n=2)),
        ("QUANT", "quantised", lambda s: lx.FusedQuantisedConfidenceLUT(s, seed=3, read_top_n=1)),
        ("QUANT", "quantised", lambda s: lx.FusedQuantisedConfidenceLUT(s, seed=3, read_top_n=2)),
        ("QUANT", "quantised", _deployed),
    ], spec


def _deployed(spec):
    p = lx.QuantisedConfidenceLUT(spec, seed=3, read_top_n=2).to_deployment()
    return lx.DeployedQuantisedConfidenceLUT(spec, p["tensors"], p["meta"], device="cuda")


@pytest.mark.parametrize("i", range(7))
@pytest.mark.parametrize("B", [5, 300])          # a partial last row block, and many CTAs per group
def test_both_forward_orders_are_bit_identical(stub_capability, monkeypatch, i, B):
    cases, kw = _cases()
    family, ext, make = cases[i]
    requires_extension(ext)
    from spiky.lutorch_ex.lut_spec import LUTSpec
    spec = LUTSpec(**kw)
    m = make(spec).cuda().eval()
    torch.manual_seed(0)
    x = torch.randn(B, spec.h_in, spec.d_in, device="cuda")
    out = {}
    for order in ("interleaved", "group"):
        monkeypatch.setenv(f"LUTORCH_EX_{family}_CUDA_CTA_ORDER", order)
        _cuda_ext._forward_group_major.cache_clear()
        with torch.no_grad():
            out[order] = m(x)
    assert torch.equal(out["interleaved"], out["group"])
