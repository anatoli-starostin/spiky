"""DominanceStackLUT: shape contract, equals reference-Dominance -> FusedManifestoHard layer by layer (value + grads)."""
import os, sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spiky.lutorch_ex import LUTSpec, ProjectionMHL, FusedManifestoHardLUT, cell_tv_penalty
from dominance_lut_ref import DominanceLUT
from stack import DominanceStackLUT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_stack_matches_layerwise_reference():
    sp = LUTSpec(h_in=8, h_out=8, tph=16, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
    second = FusedManifestoHardLUT(sp, weight="rational", seed=7, weight_init_std=0.3)
    st = DominanceStackLUT(sp, second, dom_seed=5).cuda()
    ref = DominanceLUT(sp, seed=5).cuda()
    ref.weights.data.copy_(st.dom.weights.data)
    x = torch.randn(257, 8, 48, device="cuda")
    outs = []
    for k, fn in enumerate((lambda x: st(x), lambda x: st.second(ref(x)))):
        x_ = x.clone().requires_grad_(True)
        st.zero_grad(); ref.zero_grad()
        y = fn(x_)
        (y * torch.linspace(-1, 1, y.numel(), device="cuda").reshape(y.shape)).sum().backward()
        outs.append((y.detach(), x_.grad.clone(), (st.dom if k == 0 else ref).weights.grad.clone(),
                     st.second.weights.grad.clone()))
    for a, b in zip(*outs):
        assert (a - b).abs().max() < 1e-4 * max(1.0, b.abs().max().item())
    assert y.shape == (257, 8, 48)


def test_plugs_into_projection_and_tv_sees_second_layer():
    sp = LUTSpec(h_in=8, h_out=8, tph=4, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
    st = DominanceStackLUT(sp, FusedManifestoHardLUT(sp, seed=1, table_dropout_rate=0.2)).cuda()
    m = ProjectionMHL(st, d_model=384).float().cuda()
    out = m(torch.randn(33, 384, device="cuda"))
    assert out.shape == (33, 384)
    assert cell_tv_penalty(m).numel() == 1


def test_post_norm_ln():
    import pytest
    from spiky.lutorch_ex import LUTSpec
    from spiky.lutorch_ex.cartridges import FusedManifestoHardLUT
    spec = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
    for pn, n_par in (("ln", 96), ("ln_noaffine", 0)):
        st = DominanceStackLUT(spec, FusedManifestoHardLUT(spec, weight="rational", seed=1), dom_seed=2, post_norm=pn).cuda()
        assert sum(p.numel() for p in st.norm.parameters()) == n_par
        x = torch.randn(16, 8, 48, device="cuda", requires_grad=True)
        y = st(x); y.sum().backward()
        h = st.dom(x)
        assert torch.allclose(st.norm(h).std(-1, unbiased=False), torch.ones(16, 8, device="cuda"), atol=1e-3)
        st.eval(); st(x); assert st.mean_hidden_std() > 0
