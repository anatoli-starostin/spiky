"""Quick test: flags off == the old model; flags on = fan-out single-probe no-projection geometry (CUDA needed).
Run: CART-free; `python test_noproj_geometry.py` (or pytest) with the lutorch_ex overlay on PYTHONPATH."""
import json, os, torch
from spiky.lutorch_ex import ConfidenceLUT, LUTSpec, ProjectionMHL
from noproj_geometry import make_spec, make_proj

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "lutorch_ex_conf_n1_16k_1125", "config.json")))


def _cart(spec, seed=1000):
    return ConfidenceLUT(spec, read_top_n=1, beta_init=CFG["lut_learned_margin_beta_init"],
                         gamma_init=CFG["lut_learned_margin_gamma_init"], read_tau_init=CFG["lut_read_tau"],
                         read_tau_learnable=True, seed=seed, weight_init_std=CFG["lut_init_weights_noise"],
                         table_dropout_rate=0.2)


def _old():   # verbatim the pre-flag construction
    spec = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)
    cart = _cart(spec)
    return ProjectionMHL(cart, d_model=384).float().cuda()


def test_flags_off_equals_old():
    torch.manual_seed(0); old = _old()
    torch.manual_seed(0); new = make_proj(_cart(make_spec(CFG)), 384).float().cuda()
    assert make_spec(CFG) == LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)
    so, sn = old.state_dict(), new.state_dict()
    assert so.keys() == sn.keys() and all(torch.equal(so[k], sn[k]) for k in so)
    x = torch.randn(64, 384, device="cuda")
    for m in (old, new): m.eval()
    assert torch.equal(old(x), new(x))


def test_flags_on_geometry():
    spec = make_spec(CFG, True, "single")
    assert (spec.h_in, spec.h_out, spec.d_in, spec.d_out, spec.anchor_mode) == (1, 8, 384, 48, "single")
    p = make_proj(_cart(spec), 384, True).float().cuda()
    assert isinstance(p.compress, torch.nn.Identity) and p.decompress.weight.shape == (384, 384)
    x = torch.randn(64, 384, device="cuda", requires_grad=True)
    y = p(x); assert y.shape == (64, 384); y.sum().backward()
    assert p.cartridge.weights.grad is not None


if __name__ == "__main__":
    test_flags_off_equals_old(); test_flags_on_geometry(); print("OK")
