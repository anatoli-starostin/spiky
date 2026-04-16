"""
Rank quantization module for permutational LUT architectures.

Converts real-valued LUT outputs to fixed-value rankings (STE forward).
Optionally provides rank-based backward weighting for anchor pair gradients.

Forward: raw_output -> argsort -> fixed values {0, 1, ..., N-1} / N
Backward: STE (gradients pass through as if no quantization happened)

Usage:
    rq = RankQuantize()
    output = rq(lut_output)  # [B, N] -> [B, N] with values in {0/(N-1), 1/(N-1), ..., 1}
"""
import torch
import torch.nn as nn


class RankQuantizeFunction(torch.autograd.Function):
    """
    Forward: replace values with their normalized rank {0, 1/(N-1), 2/(N-1), ..., 1}.
    Backward: straight-through estimator (pass gradient unchanged).
    """

    @staticmethod
    def forward(ctx, x):
        # x: [*, D] — last dimension is what we rank
        D = x.shape[-1]
        # argsort twice gives the rank of each element
        ranks = x.argsort(dim=-1).argsort(dim=-1).float()
        # Normalize to [0, 1] so all layers see same scale
        if D > 1:
            ranks = ranks / (D - 1)
        return ranks

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        return grad_output


class RankQuantize(nn.Module):
    """
    Replaces real-valued vectors with their rank-quantized version (STE).

    Every output dimension gets a value from {0, 1/(D-1), ..., 1} based on
    its position in the sorted order. The ranking is preserved exactly, and
    all layers see identically-spaced inputs.

    Gradients flow through via straight-through estimator.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [*, D] where D is the dimension to rank.

        Returns:
            Tensor of same shape with values replaced by normalized ranks.
        """
        if self.training:
            return RankQuantizeFunction.apply(x)
        else:
            D = x.shape[-1]
            ranks = x.argsort(dim=-1).argsort(dim=-1).float()
            if D > 1:
                ranks = ranks / (D - 1)
            return ranks


def rank_distance_weights(x: torch.Tensor, anchor_a: torch.Tensor, anchor_b: torch.Tensor,
                          eps: float = 1.0) -> torch.Tensor:
    """
    Compute gradient weights based on rank distance instead of magnitude distance.

    For each anchor pair (a, b), the weight is 1 / (|rank(x[a]) - rank(x[b])| + eps).
    Dimensions close in rank (near decision boundaries) get strong gradients.
    Dimensions far in rank get weak gradients.

    Args:
        x: Input tensor [B, D]
        anchor_a: Anchor indices [n_pairs] or [n_tables, n_anchor_pairs]
        anchor_b: Anchor indices [n_pairs] or [n_tables, n_anchor_pairs]
        eps: Minimum rank distance (default 1.0 = adjacent ranks)

    Returns:
        Weights tensor of same shape as the anchor pair distances.
    """
    D = x.shape[-1]
    # Compute ranks: [B, D]
    ranks = x.detach().argsort(dim=-1).argsort(dim=-1).float()
    # Rank distances for each anchor pair
    rank_a = ranks[:, anchor_a]  # [B, n_pairs] or [B, n_tables, nap]
    rank_b = ranks[:, anchor_b]
    rank_dist = (rank_a - rank_b).abs()
    # Weight: inversely proportional to rank distance
    weights = 1.0 / (rank_dist + eps)
    return weights
