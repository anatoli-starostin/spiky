"""ProjectionMHL with input_dim != output_dim, and the d_model alias staying backward compatible."""
import pytest
import torch
import torch.nn as nn

from spiky.lutorch_ex import (
    LUTSpec, ManifestoHardLUT, ManifestoSoftLUT, ProjectionMHL, QuantisedConfidenceLUT,
    export_deployment, load_deployment,
)

SPEC = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=5, d_out=4)   # in_features 10, out_features 8


def _proj(**kw):
    return ProjectionMHL(ManifestoSoftLUT(SPEC, seed=3), **kw)


@pytest.mark.parametrize("input_dim,output_dim", [(16, 24), (24, 16), (7, 33), (33, 7)])
def test_rectangular_shapes_and_gradients(input_dim, output_dim):
    proj = _proj(input_dim=input_dim, output_dim=output_dim).train()
    assert proj.compress.weight.shape == (SPEC.in_features, input_dim)
    assert proj.decompress.weight.shape == (output_dim, SPEC.out_features)
    x = torch.randn(6, input_dim, requires_grad=True)
    out = proj(x)
    assert out.shape == (6, output_dim)
    # decompress starts at zero, so push it off zero before checking that gradient reaches everything
    with torch.no_grad():
        proj.decompress.weight.normal_(0.0, 0.1)
    proj(x).square().sum().backward()
    assert x.grad is not None and x.grad.shape == (6, input_dim)
    assert proj.compress.weight.grad is not None and proj.compress.weight.grad.abs().sum() > 0
    assert proj.decompress.weight.grad is not None and proj.decompress.weight.grad.abs().sum() > 0
    assert proj.cartridge.weights.grad is not None and proj.cartridge.weights.grad.abs().sum() > 0


def test_forward_rejects_wrong_input_width():
    proj = _proj(input_dim=16, output_dim=24)
    with pytest.raises(ValueError):
        proj(torch.randn(2, 24))


def test_d_model_alias_positional_and_keyword_are_the_square_case():
    torch.manual_seed(0); a = ProjectionMHL(ManifestoHardLUT(SPEC, seed=1), 16)
    torch.manual_seed(0); b = ProjectionMHL(ManifestoHardLUT(SPEC, seed=1), d_model=16)
    torch.manual_seed(0); c = ProjectionMHL(ManifestoHardLUT(SPEC, seed=1), input_dim=16, output_dim=16)
    for p in (a, b, c):
        assert p.input_dim == 16 and p.output_dim == 16 and p.d_model == 16
    # identical parameter names and shapes -> checkpoints of the square form load into any spelling
    sa, sc = a.state_dict(), c.state_dict()
    assert list(sa) == list(sc) and all(sa[k].shape == sc[k].shape for k in sa)
    c.load_state_dict(sa)
    x = torch.randn(3, 16)
    assert torch.equal(a.eval()(x), c.eval()(x))


def test_d_model_must_agree_with_explicit_dims():
    _proj(d_model=16, input_dim=16, output_dim=16)          # consistent: fine
    with pytest.raises(ValueError):
        _proj(d_model=16, input_dim=12)
    with pytest.raises(ValueError):
        _proj(d_model=16, output_dim=12)
    with pytest.raises(ValueError):
        _proj(input_dim=16)                                  # output_dim missing
    with pytest.raises(ValueError):
        _proj()                                              # nothing given


def test_rectangular_has_no_single_d_model():
    proj = _proj(input_dim=16, output_dim=24)
    with pytest.raises(AttributeError):
        proj.d_model
    assert "input_dim=16" in repr(proj) and "output_dim=24" in repr(proj)


def test_side_off_width_checks_use_the_matching_dim():
    # compress off: input_dim must equal h_in*d_in (10); output_dim is free
    p = _proj(input_dim=SPEC.in_features, output_dim=24, compress=False)
    assert isinstance(p.compress, nn.Identity) and p(torch.randn(2, 10)).shape == (2, 24)
    with pytest.raises(ValueError):
        _proj(input_dim=11, output_dim=24, compress=False)
    # decompress off: output_dim must equal h_out*d_out (8); input_dim is free
    p = _proj(input_dim=24, output_dim=SPEC.out_features, decompress=False)
    assert isinstance(p.decompress, nn.Identity) and p(torch.randn(2, 24)).shape == (2, 8)
    with pytest.raises(ValueError):
        _proj(input_dim=24, output_dim=9, decompress=False)


def test_both_sides_off_is_refused_even_when_widths_would_agree():
    # With both sides off nothing can change the width, so it could only ever be the identity case
    # input_dim == output_dim == h_in*d_in == h_out*d_out -- and even then the wrapper does nothing.
    square = LUTSpec(h_in=2, h_out=2, tph=1, nap=2, d_in=8, d_out=8)   # in = out = 16
    with pytest.raises(ValueError, match="both"):
        ProjectionMHL(ManifestoHardLUT(square, seed=1), input_dim=16, output_dim=16,
                      compress=False, decompress=False)
    with pytest.raises(ValueError, match="both"):
        ProjectionMHL(ManifestoHardLUT(square, seed=1), input_dim=16, output_dim=20,
                      compress=False, decompress=False)


# ---- deployment export / load with a rectangular wrapper -------------------------------------

Q = dict(h_in=2, h_out=2, tph=4, nap=5, d_in=6, d_out=6)     # out_features = 12


class _RectModel(nn.Module):
    def __init__(self):
        super().__init__()
        common = dict(quant_mode="p2_int8", weight_init_std=1e-2, read_top_n=2,
                      beta_init=2.0, gamma_init=1.0, read_tau_init=0.5)
        self.ff = ProjectionMHL(QuantisedConfidenceLUT(LUTSpec(**Q), seed=1, **common),
                                input_dim=20, output_dim=14)

    def forward(self, x):
        return self.ff(x)


def test_deployment_roundtrip_rectangular(tmp_path):
    torch.manual_seed(0)
    m = _RectModel().eval()
    with torch.no_grad():
        m.ff.cartridge.weights.normal_(0.0, 1e-2)
        m.ff.decompress.weight.normal_(0.0, 0.1)
    path = str(tmp_path / "rect.lxq")
    meta = export_deployment(m, path)
    ff_meta = meta["ffns"]["ff"]
    assert ff_meta["input_dim"] == 20 and ff_meta["output_dim"] == 14 and "d_model" not in ff_meta

    with torch.device("meta"):
        skeleton = _RectModel()
    dep = load_deployment(path, lambda: skeleton)
    assert dep.ff.input_dim == 20 and dep.ff.output_dim == 14
    x = torch.randn(5, 20)
    ref, got = m(x), dep(x)
    assert got.shape == (5, 14)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_deployment_loader_accepts_old_square_meta(tmp_path):
    """A file written before input_dim/output_dim existed carries only "d_model"; it must still load."""
    import json
    torch.manual_seed(0)

    class _SqModel(nn.Module):
        def __init__(self):
            super().__init__()
            common = dict(quant_mode="p2_int8", weight_init_std=1e-2, read_top_n=2,
                          beta_init=2.0, gamma_init=1.0, read_tau_init=0.5)
            self.ff = ProjectionMHL(QuantisedConfidenceLUT(LUTSpec(**Q), seed=1, **common), d_model=12)

        def forward(self, x):
            return self.ff(x)

    m = _SqModel().eval()
    path = str(tmp_path / "sq.lxq")
    export_deployment(m, path)
    # rewrite the header the way an old exporter would have: d_model only
    from spiky.lutorch_ex.deploy import _load_payload, _save_payload
    tensors, meta = _load_payload(path)
    for ff in meta["ffns"].values():
        ff.pop("input_dim"); ff.pop("output_dim")
        assert ff["d_model"] == 12
    _save_payload(path, tensors, meta)
    with torch.device("meta"):
        skeleton = _SqModel()
    dep = load_deployment(path, lambda: skeleton)
    assert dep.ff.input_dim == dep.ff.output_dim == 12
    x = torch.randn(4, 12)
    assert torch.allclose(dep(x), m(x), rtol=1e-5, atol=1e-6)
