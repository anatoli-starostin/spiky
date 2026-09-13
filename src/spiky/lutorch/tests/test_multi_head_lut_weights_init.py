"""MultiHeadLut.weights_init: the default ("normal") is MultiHeadLut's historical N(0, noise^2) draw from
Generator(random_seed), unchanged; "uniform" is the FastMultiHeadLut / LightMultiHeadLUT per-head table rule
(Uniform[-noise, +noise], head h from Generator(random_seed + h + 1)), and reproduces their tables exactly."""
import pytest
import torch

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from spiky.lutorch.multi_head_lut import MultiHeadLut

D_IN, H, OUT, NAP, TPH, SEED = 6, 3, 5, 3, 4, 17


def _gen1(**kw):
    return MultiHeadLut(input_dim=H * D_IN, n_heads=H, n_outputs=OUT, n_anchor_pairs=NAP, tables_per_head=TPH,
                        random_seed=SEED, **kw)


@pytest.mark.parametrize("noise", [1e-3, 0.5])
def test_default_is_the_historical_normal_draw(noise):
    m = _gen1(initial_weights_noise=noise)
    ref = torch.randn(H * TPH, 1 << NAP, OUT, generator=torch.Generator().manual_seed(SEED)) * noise
    assert m.weights_init == "normal"
    assert torch.equal(m.projection.weights.detach(), ref)
    assert torch.equal(m.projection.weights, _gen1(initial_weights_noise=noise, weights_init="normal").projection.weights)


@pytest.mark.parametrize("noise", [1e-3, 1e-4, 0.25])
def test_uniform_reproduces_fast_and_light_tables(noise):
    w = _gen1(initial_weights_noise=noise, weights_init="uniform").projection.weights.detach()
    fast = FastMultiHeadLut(input_dim=D_IN, n_heads=H, n_outputs=OUT, n_anchor_pairs=NAP, tables_per_head=TPH,
                            multi_head_input=True, random_seed=SEED, initial_weights_noise=noise)
    light = LightMultiHeadLUT(input_dim=D_IN, n_tables=H * TPH, output_dim=OUT, n_anchor_pairs=NAP, n_heads=H,
                              multi_head_input=True, random_seed=SEED, initial_weights_noise=noise)
    assert torch.equal(w, fast.weights.detach())
    assert torch.equal(w, light.tables.detach())
    assert float(w.abs().max()) <= noise


def test_uniform_per_head_rule_and_edge_cases():
    w = _gen1(initial_weights_noise=1e-3, weights_init="uniform").projection.weights.detach()
    for h in range(H):
        u = torch.rand(TPH, 1 << NAP, OUT, generator=torch.Generator().manual_seed(SEED + h + 1)) - 0.5
        assert torch.equal(w[h * TPH:(h + 1) * TPH], u * 2e-3)
    assert torch.count_nonzero(_gen1(initial_weights_noise=0.0, weights_init="uniform").projection.weights) == 0
    unseeded = MultiHeadLut(input_dim=H * D_IN, n_heads=H, n_outputs=OUT, n_anchor_pairs=NAP, tables_per_head=TPH,
                            initial_weights_noise=1e-3, weights_init="uniform")
    assert float(unseeded.projection.weights.abs().max()) <= 1e-3
    with pytest.raises(ValueError, match="weights_init"):
        _gen1(weights_init="xavier")
