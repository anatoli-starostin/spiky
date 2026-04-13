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


class HyperLUTBackbone(nn.Module):
    """
    Shared feature backbone for HyperLUT: pairs + fc1 + GELU.

    Computes pairwise comparisons (hard forward / soft backward via STE),
    then projects to per-head hidden representations.

    Multiple HyperLUT heads can share a single backbone to amortise the
    comparison and fc1 cost.

    Args:
        input_dim:   Dimension of input vector.
        n_heads:     Number of output heads.
        n_pairs:     Number of pairwise comparisons.
        hidden_dim:  Hidden dimension per head after fc1.
        temperature: Sharpness of soft comparisons in backward.
        soft_mode:   'sigmoid' or 'rational'.
        random_seed: Seed for pair sampling.
        device:      Device for buffers and parameters.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_pairs: int,
        hidden_dim: int,
        temperature: float = 1.0,
        soft_mode: str = 'sigmoid',
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_pairs = n_pairs
        self.hidden_dim = hidden_dim
        self.soft_mode = soft_mode
        self.temperature = temperature

        gen = torch.Generator()
        if random_seed is not None:
            gen.manual_seed(random_seed)

        # Each head gets its own n_pairs comparisons
        head_pairs = []
        for _ in range(n_heads):
            head_pairs.append(_sample_pairs(input_dim, n_pairs, generator=gen))
        pairs = torch.cat(head_pairs, dim=1)  # (2, n_heads * n_pairs)
        self.register_buffer("pairs", pairs)

        # Per-head fc1: weight [n_heads, hidden_dim, n_pairs], bias [n_heads, hidden_dim]
        self.fc1_weight = nn.Parameter(torch.empty(n_heads, hidden_dim, n_pairs, device=device))
        self.fc1_bias = nn.Parameter(torch.zeros(n_heads, hidden_dim, device=device))
        nn.init.normal_(self.fc1_weight, std=0.01)

        self.norm = nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else None

    def _soft_features(self, x: torch.Tensor) -> torch.Tensor:
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]  # [B, n_heads * n_pairs]
        if self.soft_mode == 'sigmoid':
            return torch.sigmoid(d / self.temperature)
        else:
            return 0.5 + 0.5 * d / (self.temperature + d.abs())

    def _hard_features(self, x: torch.Tensor) -> torch.Tensor:
        d = x[..., self.pairs[0]] - x[..., self.pairs[1]]
        return (d > 0).float()

    def _ste_features(self, x: torch.Tensor) -> torch.Tensor:
        soft = self._soft_features(x)
        hard = self._hard_features(x)
        return (hard - soft).detach() + soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, input_dim].

        Returns:
            Per-head hidden features [B, n_heads, hidden_dim].
        """
        features = self._ste_features(x)  # [B, n_heads * n_pairs]
        features = features.view(x.shape[0], self.n_heads, self.n_pairs)  # [B, n_heads, n_pairs]
        h = torch.einsum('bhp,hdp->bhd', features, self.fc1_weight) + self.fc1_bias  # [B, n_heads, hidden_dim]
        if self.norm is not None:
            h = self.norm(h)
        return F.relu(h)


class HyperLUT(nn.Module):
    """
    Multi-head HyperLUT module.

    For each input vector x [B, input_dim]:
      1. Compute n_pairs pairwise comparisons: sign(x[a_i] - x[b_i]) → {0, 1}
         (hard forward via STE, soft backward via sigmoid surrogate)
      2. Per head: Linear(n_pairs → hidden_dim) → GELU → Linear(hidden_dim → n_outputs)

    This replaces MultiHeadLut with a fully differentiable module that has
    exact gradients (no uncertainty derivative approximation).

    Can either own its backbone (pairs + fc1) or share one via the ``backbone``
    parameter.  When a backbone is provided, ``input_dim``, ``n_heads``,
    ``n_pairs``, ``hidden_dim``, ``temperature``, ``soft_mode``, and
    ``random_seed`` are ignored (taken from the backbone).

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
        backbone:    Optional shared HyperLUTBackbone. If provided, this module
                     only owns fc2 and delegates feature extraction to the backbone.
    """

    def __init__(
        self,
        input_dim: int = 0,
        n_heads: int = 0,
        n_outputs: int = 1,
        n_pairs: int = 0,
        hidden_dim: int = 0,
        temperature: float = 1.0,
        soft_mode: str = 'sigmoid',
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        backbone: Optional[HyperLUTBackbone] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        if backbone is not None:
            self.backbone = backbone
            self.input_dim = backbone.input_dim
            self.n_heads = backbone.n_heads
            self.n_outputs = n_outputs
            self.n_pairs = backbone.n_pairs
            hidden_dim = backbone.hidden_dim
        else:
            self.backbone = HyperLUTBackbone(
                input_dim=input_dim, n_heads=n_heads, n_pairs=n_pairs,
                hidden_dim=hidden_dim, temperature=temperature,
                soft_mode=soft_mode, random_seed=random_seed, device=device,
                use_layer_norm=use_layer_norm,
            )
            self.input_dim = input_dim
            self.n_heads = n_heads
            self.n_outputs = n_outputs
            self.n_pairs = n_pairs

        self.fc2 = nn.Linear(hidden_dim, n_outputs, device=device)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, input_dim].

        Returns:
            Output tensor [B, n_heads, n_outputs].
        """
        h = self.backbone(x)  # [B, n_heads, hidden_dim]
        return self.fc2(h)    # [B, n_heads, n_outputs]
