"""
Gradient propagation test for ProjectionLUT in a folded two-layer architecture.
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.multi_head_lut import ProjectionLUT, UnfoldConfiguration
os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"


class _ToyNet(nn.Module):
    def __init__(self, device: torch.device, H: int = 8, W: int = 8, num_classes: int = 5):
        super().__init__()

        self.H = H
        self.W = W

        # First ProjectionLUT: unfold HxW, fold back to 2H x 2W
        unfold1 = UnfoldConfiguration(
            H=H,
            W=W,
            kernel_size=3,
            stride=2,
        )
        fold1 = UnfoldConfiguration(
            H=H * 2,
            W=W * 2,
            kernel_size=6,
            stride=4,
        )
        O1 = 4
        self.proj1 = ProjectionLUT(
            unfold_config=unfold1,
            n_outputs=O1,
            n_anchor_pairs=3,
            tables_per_head=2,
            initial_weights_noise=0.01,
            fold_config=fold1,
            device=device,
        )

        # Second ProjectionLUT: unfold 2H x 2W, fold back to H x W
        unfold2 = UnfoldConfiguration(
            H=H * 2,
            W=W * 2,
            kernel_size=6,
            stride=4,
        )
        fold2 = UnfoldConfiguration(
            H=H,
            W=W,
            kernel_size=3,
            stride=2,
        )
        O2 = 3
        self.proj2 = ProjectionLUT(
            unfold_config=unfold2,
            n_outputs=O2,
            n_anchor_pairs=3,
            tables_per_head=2,
            initial_weights_noise=0.01,
            fold_config=fold2,
            device=device,
        )

        self.final_layer = nn.Linear(H * W, num_classes, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W]
        x1 = self.proj1(x)  # [B, 2H, 2W]
        x2 = self.proj2(x1)  # [B, H, W]
        return self.final_layer(x2.view(x.shape[0], -1))


class _ToyNetSingle(nn.Module):
    """
    Even simpler network: single ProjectionLUT without folding.
    Used to isolate gradient issues unrelated to the folding/scatter_add path.
    """

    def __init__(self, device: torch.device, H: int = 8, W: int = 8, num_classes: int = 5):
        super().__init__()

        self.H = H
        self.W = W

        unfold = UnfoldConfiguration(
            H=H,
            W=W,
            kernel_size=3,
            stride=2,
        )
        O = 4
        self.proj = ProjectionLUT(
            unfold_config=unfold,
            n_outputs=O,
            n_anchor_pairs=3,
            tables_per_head=2,
            initial_weights_noise=0.01,
            fold_config=None,
            device=device,
        )

        # Output of ProjectionLUT without folding is [B, H_p, W_p, O].
        H_p, W_p = unfold.output_spatial_shape()
        self.final_layer = nn.Linear(H_p * W_p * O, num_classes, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W]
        x1 = self.proj(x)  # [B, H_p, W_p, O]
        return self.final_layer(x1.view(x.shape[0], -1))


class _ToyNetFoldOnly(nn.Module):
    """
    Single ProjectionLUT *with* folding, followed only by a linear layer.
    This isolates the folding/scatter_add path without a second ProjectionLUT.
    """

    def __init__(self, device: torch.device, H: int = 8, W: int = 8, num_classes: int = 5):
        super().__init__()

        self.H = H
        self.W = W

        # Same config as proj1 in _ToyNet: unfold HxW, fold back to 2H x 2W.
        unfold = UnfoldConfiguration(
            H=H,
            W=W,
            kernel_size=3,
            stride=2,
        )
        fold = UnfoldConfiguration(
            H=H * 2,
            W=W * 2,
            kernel_size=6,
            stride=4,
        )
        O = 4
        self.proj = ProjectionLUT(
            unfold_config=unfold,
            n_outputs=O,
            n_anchor_pairs=3,
            tables_per_head=2,
            initial_weights_noise=0.01,
            fold_config=fold,
            device=device,
        )

        self.final_layer = nn.Linear((H * 2) * (W * 2), num_classes, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W]
        x1 = self.proj(x)  # [B, 2H, 2W]
        return self.final_layer(x1.view(x.shape[0], -1))


def test_projection_lut_gradients_nonzero(device):
    """
    Small sanity test to ensure gradients propagate through both ProjectionLUT
    layers in a folded architecture similar to the MNIST setup.

    Uses random inputs/targets; we only check that weight gradients on both
    ProjectionLUTs are non-zero after a single backward pass.
    """
    net = _ToyNet(device=device).to(device)
    net.train()

    B = 4
    x = torch.randn(B, net.H, net.W, device=device)
    targets = torch.randint(0, net.final_layer.out_features, (B,), device=device)

    logits = net(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    grad1 = net.proj1.lut.projection.weights.grad
    grad2 = net.proj2.lut.projection.weights.grad

    assert grad1 is not None, "proj1 projection weights grad is None"
    assert grad2 is not None, "proj2 projection weights grad is None"

    mean_grad1 = grad1.abs().mean().item()
    mean_grad2 = grad2.abs().mean().item()

    # We don't require large magnitude, just non-zero to detect silent bugs.
    assert mean_grad1 > 0.0, "proj1 projection weights have zero gradient"
    assert mean_grad2 > 0.0, "proj2 projection weights have zero gradient"


def test_single_projection_lut_gradients_nonzero(device):
    """
    Gradient sanity test for a single ProjectionLUT layer without folding.

    Checks that the projection weights receive a non-zero gradient after a
    single backward pass.
    """
    net = _ToyNetSingle(device=device).to(device)
    net.train()

    B = 4
    x = torch.randn(B, net.H, net.W, device=device)
    targets = torch.randint(0, net.final_layer.out_features, (B,), device=device)

    logits = net(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    grad = net.proj.lut.projection.weights.grad
    assert grad is not None, "single-layer ProjectionLUT projection weights grad is None"
    mean_grad = grad.abs().mean().item()
    assert mean_grad > 0.0, "single-layer ProjectionLUT projection weights have zero gradient"


def test_fold_only_projection_lut_gradients_nonzero(device):
    """
    Gradient sanity test for a single ProjectionLUT layer *with folding*.

    This checks whether the folding/scatter_add path itself allows gradients
    to reach the ProjectionLUT projection weights.
    """
    net = _ToyNetFoldOnly(device=device).to(device)
    net.train()

    B = 4
    x = torch.randn(B, net.H, net.W, device=device)
    targets = torch.randint(0, net.final_layer.out_features, (B,), device=device)

    logits = net(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    grad = net.proj.lut.projection.weights.grad
    assert grad is not None, "fold-only ProjectionLUT projection weights grad is None"
    mean_grad = grad.abs().mean().item()
    assert mean_grad > 0.0, "fold-only ProjectionLUT projection weights have zero gradient"


def main() -> int:
    """
    Run ProjectionLUT gradient tests on available devices.
    """
    print("=" * 60)
    print("ProjectionLUT GRADIENT TESTS")
    print("=" * 60)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        print(f"\nTesting on {device}...")
        test_projection_lut_gradients_nonzero(torch.device(device))
        print(f"✓ Two-layer gradient test passed on {device}")

        test_single_projection_lut_gradients_nonzero(torch.device(device))
        print(f"✓ Single-layer gradient test passed on {device}")

        test_fold_only_projection_lut_gradients_nonzero(torch.device(device))
        print(f"✓ Fold-only single-layer gradient test passed on {device}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
