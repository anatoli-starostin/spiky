"""
HyperLUT: a differentiable replacement for MultiHeadLut.

Uses hard-forward / soft-backward pairwise comparisons (like RankProjection)
followed by a two-layer MLP (Linear → GELU → Linear) to produce per-head outputs.

The "lookup table" is implicit: the first linear layer maps binary comparison
results to a hidden representation, and the second projects to outputs.
This is equivalent to a soft interpolation over 2^M table entries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _sample_pairs(
    input_dim: int,
    n_pairs: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample n_pairs unique (i, j) pairs from upper triangle of input_dim."""
    max_pairs = input_dim * (input_dim - 1) // 2
    if n_pairs > max_pairs:
        raise ValueError(f"n_pairs={n_pairs} exceeds max {max_pairs} for input_dim={input_dim}")
    i, j = torch.triu_indices(input_dim, input_dim, offset=1)
    perm = torch.randperm(max_pairs, generator=generator)[:n_pairs]
    return torch.stack([i[perm], j[perm]])  # (2, n_pairs)


class HyperLUT(nn.Module):
    """
    Multi-head HyperLUT module.

    For each input vector x [B, input_dim]:
      1. Compute n_pairs pairwise comparisons: sign(x[a_i] - x[b_i]) → {0, 1}
         (hard forward via STE, soft backward via sigmoid surrogate)
      2. Per head: Linear(n_pairs → hidden_dim) → GELU → Linear(hidden_dim → n_outputs)

    This replaces MultiHeadLut with a fully differentiable module that has
    exact gradients (no uncertainty derivative approximation).

    Args:
        input_dim:   Dimension of input vector.
        n_heads:     Number of output heads.
        n_outputs:   Output dimension per head.
        n_pairs:     Number of pairwise comparisons (like n_anchor_pairs * tables_per_head).
        hidden_dim:  Hidden dimension of the MLP.
        temperature: Sharpness of soft comparisons in backward (lower → sharper).
        soft_mode:   'sigmoid' or 'rational'. sigmoid: sigmoid(d/t). rational: 0.5 + 0.5*d/(t+|d|).
        random_seed: Seed for pair sampling.
        device:      Device for buffers and parameters.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_pairs: int,
        hidden_dim: int,
        temperature: float = 1.0,
        soft_mode: str = 'sigmoid',
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_pairs = n_pairs
        self.soft_mode = soft_mode
        self.temperature = temperature

        gen = torch.Generator()
        if random_seed is not None:
            gen.manual_seed(random_seed)

        pairs = _sample_pairs(input_dim, n_pairs, generator=gen)
        self.register_buffer("pairs", pairs)  # (2, n_pairs)

        # Per-head MLP: shared pair features → per-head hidden → per-head output
        # Using a single batched linear for efficiency
        self.fc1 = nn.Linear(n_pairs, n_heads * hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, n_outputs, device=device)

        # Initialize with small weights
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def _soft_features(self, x: torch.Tensor) -> torch.Tensor:
        """Soft comparison features → (0, 1)."""
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]  # (..., n_pairs)
        if self.soft_mode == 'sigmoid':
            return torch.sigmoid(d / self.temperature)
        else:  # rational
            return 0.5 + 0.5 * d / (self.temperature + d.abs())

    def _hard_features(self, x: torch.Tensor) -> torch.Tensor:
        """Hard comparison features: (x[a] > x[b]).float() → {0, 1}."""
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]
        return (d > 0).float()

    def _ste_features(self, x: torch.Tensor) -> torch.Tensor:
        """Hard forward, soft backward via straight-through estimator."""
        soft = self._soft_features(x)
        hard = self._hard_features(x)
        return (hard - soft).detach() + soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, input_dim].

        Returns:
            Output tensor [B, n_heads, n_outputs].
        """
        B = x.shape[0]

        # Pairwise comparisons: [B, n_pairs]
        features = self._ste_features(x)

        # Project to per-head hidden: [B, n_heads * hidden_dim]
        h = self.fc1(features)
        h = h.view(B, self.n_heads, -1)  # [B, n_heads, hidden_dim]
        h = F.gelu(h)

        # Project to outputs: [B, n_heads, n_outputs]
        out = self.fc2(h)

        return out
