"""
Anchor pairs lookup implementation.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from spiky.lutorch.abstract_lookup import AbstractLookup
from spiky.lutorch.anchor_sampler import AnchorSampler
from spiky.lutorch.lut_helpers import UncertaintyMode, get_balanced_anchor_pairs
from spiky.util.chunk_of_connections import ChunkOfConnections


class AnchorPairsLookup(AbstractLookup):
    """
    Lookup based on anchor pairs comparison.
    
    Each table uses anchor pairs to form a binary representation:
    - For each anchor pair (a1, a2), compute delta = x[a1] - x[a2]
    - If delta > 0, set bit to 1, else 0
    - The lookup index is the binary number formed by these bits
    
    Args:
        input_dim: Dimension of input tensor
        n_tables: Number of lookup tables
        n_anchor_pairs: Number of anchor pairs per table
        connected_pairs: If True, anchor pairs form a connected graph
        anchor_candidates: Optional. Either:
                          - torch.Tensor: Shape [n_tables, max_anchors_per_table] with input indices
                            (all values must be >= 0, no padding)
                          - Tuple[ChunkOfConnections, int]: ChunkOfConnections with custom ids_shift
                          - None: Uses all input indices (default)
        cmp_eps: Epsilon for comparison (default: 0.0)
        random_seed: Random seed for anchor pair sampling
        n_alternatives: Number of alternative lookup indices per table (default: 1)
                        Must be <= n_anchor_pairs. Alternatives are created by flipping bits
                        at positions corresponding to anchor pairs with minimal absolute deltas.
        anchor_initialization: "default" = use AnchorSampler (uniform/connected); "balanced" =
                        match spike_QK: indices from concatenated randperms for even dimension coverage.
    """

    def __init__(
        self,
        input_dim: int,
        n_tables: int,
        n_anchor_pairs: int,
        connected_pairs: bool = False,
        anchor_candidates: Optional[Union[torch.Tensor, Tuple[ChunkOfConnections, int]]] = None,
        cmp_eps: float = 0.0,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        n_alternatives: int = 1,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
        anchor_initialization: str = "default",
    ):
        table_dim = 2 ** n_anchor_pairs
        if n_alternatives > n_anchor_pairs:
            raise ValueError(
                f"n_alternatives ({n_alternatives}) must be <= n_anchor_pairs ({n_anchor_pairs})"
            )
        super().__init__(input_dim, n_tables, table_dim, n_alternatives=n_alternatives)

        self.n_anchor_pairs = n_anchor_pairs
        self.connected_pairs = connected_pairs
        assert cmp_eps >= 0.0
        self.cmp_eps = cmp_eps
        self.uncertainty_mode = uncertainty_mode

        dev = device or torch.device("cpu")
        if anchor_initialization == "balanced":
            # Match spike_QK: balanced coverage over input dimensions (randperm-based)
            anchor_pairs_a, anchor_pairs_b = get_balanced_anchor_pairs(
                n_tables, n_anchor_pairs, input_dim, dev, random_seed=random_seed
            )
            self.register_buffer("anchor_pairs_a", anchor_pairs_a.contiguous())
            self.register_buffer("anchor_pairs_b", anchor_pairs_b.contiguous())
        else:
            # Use AnchorSampler to sample anchor pairs
            anchor_sampler = AnchorSampler(
                n_inputs=input_dim,
                n_detectors=n_tables,
                n_anchors_per_detector=n_anchor_pairs,
                connected_anchors_mode=connected_pairs,
                device=dev,
                detector_connections=anchor_candidates,
                compact_mode=True,
                random_seed=random_seed
            )
            anchor_pairs = anchor_sampler.get_anchor_pairs().to(dtype=torch.long)  # [n_tables, n_anchor_pairs, 2]
            self.register_buffer('anchor_pairs_a', anchor_pairs[:, :, 0].contiguous())
            self.register_buffer('anchor_pairs_b', anchor_pairs[:, :, 1].contiguous())

        # Pre-compute powers tensor for bit shifting: [1, 1, n_anchor_pairs]
        powers = torch.arange(n_anchor_pairs, dtype=torch.long).view(1, 1, -1)
        if device is not None:
            powers = powers.to(device)
        self.register_buffer('powers', powers)

        # Cache for batch_offset in backward; recomputed when batch_size changes
        self._cached_batch_offset = None

    def forward(
        self,
        x: torch.Tensor,
        return_alternatives=True
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, input_dim]
            return_alternatives: in eval mode can be set to False
            
        Returns:
            In training mode:
                - lookup_indices: int64 [B, n_tables]
                - lookup_alt_indices: int64 [B, n_tables, n_alternatives]
                  Ordered by ascending absolute delta (smallest first)
                - lookup_alt_deltas: float [B, n_tables, n_alternatives]
                  Ordered by ascending absolute delta (smallest first)
                - lookup_indices_grad_c: float [B, n_tables]
                - lookup_alt_indices_grad_c: float [B, n_tables, n_alternatives]
            In eval mode:
                - lookup_indices: int64 [B, n_tables]
                - lookup_alt_indices: int64 [B, n_tables, n_alternatives] (or empty if return_alternatives=False)
                  Ordered by ascending absolute delta (smallest first)
        """
        batch_size = x.shape[0]
        device = x.device

        # Check that module buffers are on the same device as input
        assert self.anchor_pairs_a.device == device, \
            f"Module buffers device ({self.anchor_pairs_a.device}) must match input device ({device})"

        # Get anchor pairs as separate tensors
        anchor_pairs_a = self.anchor_pairs_a  # [n_tables, n_anchor_pairs]
        anchor_pairs_b = self.anchor_pairs_b  # [n_tables, n_anchor_pairs]

        if self.training:
            assert return_alternatives
            return self._forward_train(x, anchor_pairs_a, anchor_pairs_b)
        else:
            return self._forward_eval(x, anchor_pairs_a, anchor_pairs_b, return_alternatives)

    def _forward_eval(
        self,
        x: torch.Tensor,
        anchor_pairs_a: torch.Tensor,
        anchor_pairs_b: torch.Tensor,
        return_alternatives: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluation forward pass.
        
        Returns:
            - lookup_indices: int64 [B, n_tables]
            - lookup_alt_indices: int64 [B, n_tables, n_alternatives] (or None if return_alternatives=False)
            - lookup_alt_deltas: float [B, n_tables, n_alternatives] (or None if return_alternatives=False)
        """
        # anchor_pairs_a/b: [n_tables, n_anchor_pairs] -> gather on flattened index
        batch_size = x.shape[0]
        idx_a = anchor_pairs_a.reshape(1, -1).expand(batch_size, -1)  # [B, n_tables*n_anchor_pairs]
        idx_b = anchor_pairs_b.reshape(1, -1).expand(batch_size, -1)
        x_a = x.gather(1, idx_a).view(batch_size, self.n_tables, self.n_anchor_pairs)
        x_b = x.gather(1, idx_b).view(batch_size, self.n_tables, self.n_anchor_pairs)
        deltas = x_a - x_b

        # Form binary representation: [B, n_tables, n_anchor_pairs]
        bits = deltas.gt(self.cmp_eps).to(dtype=torch.long)

        # Convert to integer lookup index: [B, n_tables]
        # lookup_index = sum(bits[i] << i) for each table
        lookup_indices = (bits << self.powers).sum(dim=2, dtype=torch.long)  # [B, n_tables]

        # Compute alternative indices by flipping bits at positions with minimal absolute deltas
        if return_alternatives:
            abs_deltas = deltas.abs()  # [B, n_tables, n_anchor_pairs]
            if self.n_alternatives == 1:
                _, min_delta_indices = abs_deltas.min(dim=2, keepdim=True)
                lookup_alt_deltas = deltas.gather(2, min_delta_indices)
            else:
                min_delta_indices = torch.topk(
                    abs_deltas, k=self.n_alternatives, dim=2, largest=False
                ).indices
                lookup_alt_deltas = deltas.gather(2, min_delta_indices)
            lookup_indices_expanded = lookup_indices.unsqueeze(2)
            flip_masks = (1 << min_delta_indices).long()
            lookup_alt_indices = (lookup_indices_expanded ^ flip_masks)
        else:
            lookup_alt_indices = None
            lookup_alt_deltas = None
        
        return lookup_indices, lookup_alt_indices, lookup_alt_deltas

    def _forward_train(
        self,
        x: torch.Tensor,
        anchor_pairs_a: torch.Tensor,
        anchor_pairs_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass with gradient carriers."""
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        expected_len = batch_size * self.n_tables * self.n_alternatives
        if (
            self._cached_batch_offset is None
            or self._cached_batch_offset.numel() != expected_len
            or self._cached_batch_offset.device != x.device
        ):
            self._cached_batch_offset = (
                torch.arange(batch_size, device=x.device, dtype=torch.long)
                .repeat_interleave(self.n_tables * self.n_alternatives) * input_dim
            ).contiguous()
        uncertainty_mode_int = 0 if self.uncertainty_mode == UncertaintyMode.INVERSE_L1 else 1
        return AnchorPairsLookupFunction.apply(
            x, anchor_pairs_a, anchor_pairs_b, self.powers, self.cmp_eps,
            uncertainty_mode_int, self.n_alternatives, self._cached_batch_offset
        )


class AnchorPairsLookupFunction(torch.autograd.Function):
    """Custom autograd function for anchor pairs lookup with gradient propagation."""

    @staticmethod
    @torch.compile(dynamic=True)
    def forward(ctx, *args):
        """
        Forward pass.

        Args:
            ctx: Context object
            *args: x, anchor_pairs_a, anchor_pairs_b, powers, cmp_eps,
                   uncertainty_mode (Python int), n_alternatives (Python int), batch_offset (tensor)

        Returns:
            lookup_indices, lookup_alt_indices, lookup_alt_deltas,
            lookup_indices_grad_c, lookup_alt_indices_grad_c
        """
        (
            x, anchor_pairs_a, anchor_pairs_b,
            powers, cmp_eps, uncertainty_mode, n_alternatives, batch_offset
        ) = args
        batch_size = x.shape[0]
        n_anchor_pairs = powers.shape[-1]
        n_tables = anchor_pairs_a.shape[0]
        idx_a = anchor_pairs_a.reshape(1, -1).expand(batch_size, -1)  # [B, n_tables*n_anchor_pairs]
        idx_b = anchor_pairs_b.reshape(1, -1).expand(batch_size, -1)
        x_a = x.gather(1, idx_a).view(batch_size, n_tables, n_anchor_pairs)
        x_b = x.gather(1, idx_b).view(batch_size, n_tables, n_anchor_pairs)
        deltas = x_a - x_b

        bits = deltas.gt(cmp_eps).long()
        lookup_indices = (bits << powers).sum(dim=2, dtype=torch.long)  # [B, n_tables]

        abs_deltas = deltas.abs()  # [B, n_tables, n_anchor_pairs]
        if n_alternatives == 1:
            _, min_delta_indices = abs_deltas.min(dim=2, keepdim=True)  # [B, n_tables, 1]
            lookup_alt_deltas = deltas.gather(2, min_delta_indices)
        else:
            min_delta_indices = torch.topk(
                abs_deltas, k=n_alternatives, dim=2, largest=False
            ).indices  # [B, n_tables, n_alternatives]
            lookup_alt_deltas = deltas.gather(2, min_delta_indices)

        lookup_indices_expanded = lookup_indices.unsqueeze(2)
        flip_masks = (1 << min_delta_indices).long()
        lookup_alt_indices = (lookup_indices_expanded ^ flip_masks)

        # Per (B, table, alt) take that table's row at that pair-index
        anchor1_ids = anchor_pairs_a.unsqueeze(0).expand(batch_size, -1, -1).gather(2, min_delta_indices)
        anchor2_ids = anchor_pairs_b.unsqueeze(0).expand(batch_size, -1, -1).gather(2, min_delta_indices)

        z = x.sum() * 0
        lookup_indices_grad_c = z.expand(batch_size, n_tables)
        lookup_alt_indices_grad_c = z.expand(batch_size, n_tables, n_alternatives)

        ctx.inv_l1 = (int(uncertainty_mode) == 0)
        ctx.batch_offset = batch_offset.to(x.device).long().contiguous()
        ctx.save_for_backward(x, anchor1_ids, anchor2_ids, lookup_alt_deltas)

        return (
            lookup_indices,
            lookup_alt_indices,
            lookup_alt_deltas,
            lookup_indices_grad_c,
            lookup_alt_indices_grad_c
        )

    @staticmethod
    @torch.compile(dynamic=True)
    def backward(ctx, *grad_outputs):
        """Backward pass: propagates gradients through the anchor pairs using the uncertainty function."""
        (
            _,
            _,
            _,
            grad_lookup_indices_grad_c,
            grad_lookup_alt_indices_grad_c
        ) = grad_outputs

        x, anchor1_ids, anchor2_ids, lookup_alt_deltas = ctx.saved_tensors
        batch_size, input_dim = x.shape[0], x.shape[1]

        grad_main = grad_lookup_indices_grad_c
        grad_alt = grad_lookup_alt_indices_grad_c
        grad_diff = grad_main.unsqueeze(2) - grad_alt

        if ctx.inv_l1:
            abs_delta = lookup_alt_deltas.abs()
            one_plus_abs = 1.0 + abs_delta
            minus_uncertainty_derivative = 0.5 * lookup_alt_deltas.sign() / (one_plus_abs * one_plus_abs)
        else:
            delta_sq = lookup_alt_deltas * lookup_alt_deltas
            one_plus_sq = 1.0 + delta_sq
            minus_uncertainty_derivative = lookup_alt_deltas / (one_plus_sq * one_plus_sq)

        du = grad_diff * minus_uncertainty_derivative  # [B, n_tables, n_alternatives]

        batch_offset = ctx.batch_offset
        anchor1_flat = anchor1_ids.view(-1)
        anchor2_flat = anchor2_ids.view(-1)
        du_flat = du.view(-1)
        x_grad_flat = torch.zeros(batch_size * input_dim, device=x.device, dtype=x.dtype)
        indices1 = batch_offset + anchor1_flat
        indices2 = batch_offset + anchor2_flat
        x_grad_flat.scatter_add_(0, indices1, du_flat)
        x_grad_flat.scatter_add_(0, indices2, -du_flat)

        # 8 inputs -> 8 gradient returns
        return (x_grad_flat.view(batch_size, input_dim), None, None, None, None, None, None, None)
