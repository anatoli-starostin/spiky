"""Gradient propagation tests for ProjectionLUT in folded two-layer architectures."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from spiky.lutorch.multi_head_lut import ProjectionLUT, UnfoldConfiguration


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_torch_generator(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


class _ToyNet(nn.Module):
    def __init__(self, device, H: int = 8, W: int = 8, num_classes: int = 5,
                 random_seed: int = 1234):
        super().__init__()
        self.H, self.W = H, W
        O1 = 4
        self.proj1 = ProjectionLUT(
            unfold_config=UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=2),
            n_outputs=O1, n_anchor_pairs=3, tables_per_head=2,
            random_seed=random_seed, initial_weights_noise=0.01,
            fold_config=UnfoldConfiguration(H=H*2, W=W*2, kernel_size=6, stride=4),
            device=device,
        )
        O2 = 3
        self.proj2 = ProjectionLUT(
            unfold_config=UnfoldConfiguration(H=H*2, W=W*2, kernel_size=6, stride=4),
            n_outputs=O2, n_anchor_pairs=3, tables_per_head=2,
            random_seed=random_seed + 1, initial_weights_noise=0.01,
            fold_config=UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=2),
            device=device,
        )
        self.final_layer = nn.Linear(H * W, num_classes, device=device)

    def forward(self, x):
        return self.final_layer(self.proj2(self.proj1(x)).view(x.shape[0], -1))


class _ToyNetSingle(nn.Module):
    """Single ProjectionLUT without folding."""

    def __init__(self, device, H: int = 8, W: int = 8, num_classes: int = 5,
                 random_seed: int = 1234):
        super().__init__()
        self.H, self.W = H, W
        unfold = UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=2)
        O = 4
        self.proj = ProjectionLUT(
            unfold_config=unfold, n_outputs=O, n_anchor_pairs=3, tables_per_head=2,
            random_seed=random_seed, initial_weights_noise=0.01,
            fold_config=None, device=device,
        )
        H_p, W_p = unfold.output_spatial_shape()
        self.final_layer = nn.Linear(H_p * W_p * O, num_classes, device=device)

    def forward(self, x):
        return self.final_layer(self.proj(x).view(x.shape[0], -1))


class _ToyNetFoldOnly(nn.Module):
    """Single ProjectionLUT with folding."""

    def __init__(self, device, H: int = 8, W: int = 8, num_classes: int = 5,
                 random_seed: int = 1234):
        super().__init__()
        self.H, self.W = H, W
        self.proj = ProjectionLUT(
            unfold_config=UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=2),
            n_outputs=4, n_anchor_pairs=3, tables_per_head=2,
            random_seed=random_seed, initial_weights_noise=0.01,
            fold_config=UnfoldConfiguration(H=H*2, W=W*2, kernel_size=6, stride=4),
            device=device,
        )
        self.final_layer = nn.Linear((H*2) * (W*2), num_classes, device=device)

    def forward(self, x):
        return self.final_layer(self.proj(x).view(x.shape[0], -1))


def test_projection_lut_gradients_nonzero(device):
    """Both ProjectionLUT layers in a folded two-layer architecture receive non-zero gradients."""
    _set_seed(1234)
    gen = _make_torch_generator(1234, device)
    net = _ToyNet(device=device).to(device).train()

    x = torch.randn(4, net.H, net.W, device=device, generator=gen)
    targets = torch.randint(0, net.final_layer.out_features, (4,), device=device, generator=gen)
    F.cross_entropy(net(x), targets).backward()

    grad1 = net.proj1.lut.projection.weights.grad
    grad2 = net.proj2.lut.projection.weights.grad
    assert grad1 is not None, "proj1 projection weights grad is None"
    assert grad2 is not None, "proj2 projection weights grad is None"
    assert grad1.abs().mean().item() > 0.0, "proj1 projection weights have zero gradient"
    assert grad2.abs().mean().item() > 0.0, "proj2 projection weights have zero gradient"


def test_single_projection_lut_gradients_nonzero(device):
    """Single ProjectionLUT without folding receives non-zero gradients."""
    _set_seed(1234)
    gen = _make_torch_generator(1234, device)
    net = _ToyNetSingle(device=device).to(device).train()

    x = torch.randn(4, net.H, net.W, device=device, generator=gen)
    targets = torch.randint(0, net.final_layer.out_features, (4,), device=device, generator=gen)
    F.cross_entropy(net(x), targets).backward()

    grad = net.proj.lut.projection.weights.grad
    assert grad is not None, "single-layer ProjectionLUT projection weights grad is None"
    assert grad.abs().mean().item() > 0.0, "single-layer ProjectionLUT projection weights have zero gradient"


def test_fold_only_projection_lut_gradients_nonzero(device):
    """Single ProjectionLUT with folding (scatter_add path) receives non-zero gradients."""
    _set_seed(1234)
    gen = _make_torch_generator(1234, device)
    net = _ToyNetFoldOnly(device=device).to(device).train()

    x = torch.randn(4, net.H, net.W, device=device, generator=gen)
    targets = torch.randint(0, net.final_layer.out_features, (4,), device=device, generator=gen)
    F.cross_entropy(net(x), targets).backward()

    grad = net.proj.lut.projection.weights.grad
    assert grad is not None, "fold-only ProjectionLUT projection weights grad is None"
    assert grad.abs().mean().item() > 0.0, "fold-only ProjectionLUT projection weights have zero gradient"
