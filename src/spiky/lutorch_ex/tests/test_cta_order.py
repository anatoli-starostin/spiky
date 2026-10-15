"""The fused kernels' CTA order: the backward kernels run their CTAs group-major (one group's CTAs consecutive: the
gradient atomics in flight stay within one or two groups' slices, in L2), the forward kernels interleaved
(g = blockIdx.x % G).

A source check pins both orders, so neither path changes silently. On a CUDA device, every row's forward output and
input gradient are written by the one CTA that owns the row: they must not depend on the batch around them, so the
first rows of a 300-row batch (many CTAs per group) equal the same rows run as a 5-row batch (a partial row block).
The forward bit for bit. The input gradient to float32 re-association only: some backward kernels (FusedManifestoSoft,
FusedConfidenceLUT) sum it with float atomics inside the CTA, so it differs by ~1e-9 even between two runs of the
same batch (with either CTA order)."""
import re
from pathlib import Path

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.lut_spec import LUTSpec
from spiky.lutorch_ex.tests._requires_extension import requires_extension

CSRC = Path(__file__).resolve().parents[1] / "cartridges" / "csrc"
_OWNER = re.compile(r"(\w+_(?:fwd|bwd)_(?:body|kernel)\w*)\s*\(")
_FWD = re.compile(r"const int g = blockIdx\.x % p\.G;\n\s*const int b0 = static_cast<int>\(\(static_cast<int64_t>"
                  r"\(blockIdx\.x\) / p\.G\) \* [\w:.]+\);")
_BWD = re.compile(r"const int64_t cta = static_cast<int64_t>\(blockIdx\.x\);\n\s*const int64_t nb = \(p\.B \+ ([\w:.]+) - 1\)"
                  r" / \1;.*\n\s*const int g = static_cast<int>\(cta / nb\);\n\s*const int b0 = static_cast<int>"
                  r"\(\(cta % nb\) \* \1\);")


def test_backward_sites_are_group_major_and_forward_sites_interleaved():
    seen = {"fwd": [], "bwd": []}
    for src in sorted(CSRC.glob("*.cu")):
        s = src.read_text()
        for pat, kind in ((_FWD, "fwd"), (_BWD, "bwd")):
            for m in pat.finditer(s):
                owner = _OWNER.findall(s[:m.start()])[-1]
                assert f"_{kind}_" in owner, f"{src.name}: {owner} has the {kind} CTA order"
                seen[kind].append(owner)
    # manifesto / confidence / quant / deploy forwards; manifesto / confidence / quant backwards
    assert len(seen["fwd"]) == 4 and len(seen["bwd"]) == 3, seen


def _cases():
    return [
        ("manifesto", lambda s: lx.FusedManifestoHardLUT(s, seed=3), True),
        ("manifesto", lambda s: lx.FusedManifestoSoftLUT(s, seed=3), True),
        ("confidence", lambda s: lx.FusedConfidenceLUT(s, seed=3, read_top_n=1), True),
        ("confidence", lambda s: lx.FusedConfidenceLUT(s, seed=3, read_top_n=2), True),
        ("quantised", lambda s: lx.FusedQuantisedConfidenceLUT(s, seed=3, read_top_n=1), True),
        ("quantised", lambda s: lx.FusedQuantisedConfidenceLUT(s, seed=3, read_top_n=2), True),
        ("quantised", _deployed, False),
    ]


def _deployed(spec):
    p = lx.QuantisedConfidenceLUT(spec, seed=3, read_top_n=2).to_deployment()
    return lx.DeployedQuantisedConfidenceLUT(spec, p["tensors"], p["meta"], device="cuda")


@pytest.mark.parametrize("i", range(7))
def test_rows_do_not_depend_on_the_batch_around_them(i):
    ext, make, trains = _cases()[i]
    requires_extension(ext)
    spec = LUTSpec(h_in=4, h_out=4, tph=16, nap=6, d_in=24, d_out=24)
    m = make(spec).cuda()
    torch.manual_seed(0)
    x = torch.randn(300, spec.h_in, spec.d_in, device="cuda")
    go = torch.randn(300, spec.h_out, spec.d_out, device="cuda")
    out = {}
    for B in (300, 5):
        if trains:
            m.train()
            xb = x[:B].clone().requires_grad_(True)
            y = m(xb)
            y.backward(go[:B])
            out[B] = (y.detach()[:5], xb.grad[:5])
        else:
            with torch.no_grad():
                out[B] = (m.eval()(x[:B])[:5],)
    assert torch.equal(out[300][0], out[5][0]), "forward rows depend on the batch around them"
    if trains:
        gz, gz5 = out[300][1], out[5][1]
        assert (gz - gz5).abs().max() <= 1e-6 * gz.abs().max(), "input-gradient rows depend on the batch around them"
