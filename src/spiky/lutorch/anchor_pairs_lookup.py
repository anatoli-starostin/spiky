"""
Anchor pairs lookup implementation.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from abstract_lookup import AbstractLookup
from anchor_sampler import AnchorSampler
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
        n_alternatives: int = 1
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

        # Use AnchorSampler to sample anchor pairs
        # AnchorSampler now supports both tensor format (anchor_candidates) and ChunkOfConnections
        anchor_sampler = AnchorSampler(
            n_inputs=input_dim,
            n_detectors=n_tables,
            n_anchors_per_detector=n_anchor_pairs,
            connected_anchors_mode=connected_pairs,
            device=device,
            detector_connections=anchor_candidates,  # Can be tensor, ChunkOfConnections, or None
            compact_mode=True,
            random_seed=random_seed
        )
        
        # Get anchor pairs from sampler and split into two separate tensors
        anchor_pairs = anchor_sampler.get_anchor_pairs()  # [n_tables, n_anchor_pairs, 2]
        # Split into two tensors: [n_tables, n_anchor_pairs] each
        self.register_buffer('anchor_pairs_a', anchor_pairs[:, :, 0].contiguous())
        self.register_buffer('anchor_pairs_b', anchor_pairs[:, :, 1].contiguous())
        
        # Pre-compute powers tensor for bit shifting: [1, 1, n_anchor_pairs]
        powers = torch.arange(n_anchor_pairs, dtype=torch.int32).view(1, 1, -1)
        if device is not None:
            powers = powers.to(device)
        self.register_buffer('powers', powers)

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
                - lookup_indices: int32 [B, n_tables]
                - lookup_alt_indices: int32 [B, n_tables, n_alternatives]
                  Ordered by ascending absolute delta (smallest first)
                - lookup_alt_deltas: float [B, n_tables, n_alternatives]
                  Ordered by ascending absolute delta (smallest first)
                - lookup_indices_grad_c: float [B, n_tables]
                - lookup_alt_indices_grad_c: float [B, n_tables, n_alternatives]
            In eval mode:
                - lookup_indices: int32 [B, n_tables]
                - lookup_alt_indices: int32 [B, n_tables, n_alternatives] (or empty if return_alternatives=False)
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
            return self._forward_train(x, anchor_pairs_a, anchor_pairs_b, batch_size)
        else:
            return self._forward_eval(x, anchor_pairs_a, anchor_pairs_b, return_alternatives, batch_size, device)
    
    def _forward_eval(
        self,
        x: torch.Tensor,
        anchor_pairs_a: torch.Tensor,
        anchor_pairs_b: torch.Tensor,
        return_alternatives: bool,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation forward pass."""
        # anchor_pairs_a: [n_tables, n_anchor_pairs]
        # anchor_pairs_b: [n_tables, n_anchor_pairs]
        
        # Use torch.gather to get anchor values for all tables at once
        # x: [B, input_dim]
        # We need: [B, n_tables, n_anchor_pairs]
        # Expand x: [B, 1, input_dim] then expand to [B, n_tables, input_dim]
        x_expanded = x.unsqueeze(1).expand(batch_size, self.n_tables, x.shape[1])  # [B, n_tables, input_dim]
        
        # Expand anchor IDs to match batch dimension: [B, n_tables, n_anchor_pairs]
        anchor1_ids_expanded = anchor_pairs_a.unsqueeze(0).expand(batch_size, self.n_tables, self.n_anchor_pairs)  # [B, n_tables, n_anchor_pairs]
        anchor2_ids_expanded = anchor_pairs_b.unsqueeze(0).expand(batch_size, self.n_tables, self.n_anchor_pairs)  # [B, n_tables, n_anchor_pairs]
        
        # Gather anchor1 values: [B, n_tables, n_anchor_pairs]
        x_anchor1 = torch.gather(x_expanded, dim=2, index=anchor1_ids_expanded)
        
        # Gather anchor2 values: [B, n_tables, n_anchor_pairs]
        x_anchor2 = torch.gather(x_expanded, dim=2, index=anchor2_ids_expanded)
        
        # Compute deltas: [B, n_tables, n_anchor_pairs]
        deltas = x_anchor1 - x_anchor2
        
        # Form binary representation: [B, n_tables, n_anchor_pairs]
        bits = deltas.gt(self.cmp_eps).int_()
        
        # Convert to integer lookup index: [B, n_tables]
        # lookup_index = sum(bits[i] << i) for each table
        lookup_indices = (bits << self.powers).sum(dim=2, dtype=torch.int32)  # [B, n_tables]

        # Compute alternative indices by flipping bits at positions with minimal absolute deltas
        if return_alternatives:
            # Find top K minimal absolute deltas: [B, n_tables, n_alternatives]
            # Results are sorted in ascending order by absolute delta (smallest first)
            abs_deltas = deltas.abs()  # [B, n_tables, n_anchor_pairs]
            # Get top K smallest deltas and their indices
            min_delta_indices = torch.topk(
                abs_deltas, k=self.n_alternatives, dim=2, largest=False
            ).indices  # [B, n_tables, n_alternatives] each, sorted by ascending absolute delta

            # Create alternative indices by flipping bits at minimal delta positions
            # For each alternative, flip the bit at the corresponding anchor pair position
            # alt_index = lookup_index ^ (1 << min_delta_index)
            lookup_indices_expanded = lookup_indices.unsqueeze(2)  # [B, n_tables, 1] for broadcasting
            flip_masks = (1 << min_delta_indices).int()  # [B, n_tables, n_alternatives]
            lookup_alt_indices = (lookup_indices_expanded ^ flip_masks)  # [B, n_tables, n_alternatives]
        else:
            lookup_alt_indices = torch.empty((batch_size, self.n_tables, 0), dtype=torch.int32, device=device)
        
        return lookup_indices, lookup_alt_indices
    
    def _forward_train(
        self,
        x: torch.Tensor,
        anchor_pairs_a: torch.Tensor,
        anchor_pairs_b: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass with gradient carriers."""
        # Use autograd function for custom backward
        return AnchorPairsLookupFunction.apply(
            x, anchor_pairs_a, anchor_pairs_b, self.powers, self.cmp_eps,
            batch_size, self.n_tables, self.n_alternatives
        )


class AnchorPairsLookupFunction(torch.autograd.Function):
    """Custom autograd function for anchor pairs lookup with gradient propagation."""
    
    @staticmethod
    def forward(ctx, *args):
        """
        Forward pass.
        
        Args:
            ctx: Context object
            *args: x, anchor_pairs_a, anchor_pairs_b, powers, cmp_eps, batch_size, n_tables, n_alternatives
        
        Returns:
            lookup_indices: int32 [B, n_tables]
            lookup_alt_indices: int32 [B, n_tables, n_alternatives]
              Ordered by ascending absolute delta (smallest first)
            lookup_alt_deltas: float [B, n_tables, n_alternatives]
              Ordered by ascending absolute delta (smallest first)
            lookup_indices_grad_c: float [B, n_tables]
            lookup_alt_indices_grad_c: float [B, n_tables, n_alternatives]
        """
        x, anchor_pairs_a, anchor_pairs_b, powers, cmp_eps, batch_size, n_tables, n_alternatives = args
        n_anchor_pairs = powers.shape[-1]
        # anchor_pairs_a: [n_tables, n_anchor_pairs]
        # anchor_pairs_b: [n_tables, n_anchor_pairs]
        
        # Use advanced indexing to get anchor values for all tables at once
        # x: [B, input_dim]
        # We need: [B, n_tables, n_anchor_pairs]
        # Expand x: [B, 1, input_dim] then expand to [B, n_tables, input_dim]
        x_expanded = x.unsqueeze(1).expand(batch_size, n_tables, x.shape[1])  # [B, n_tables, input_dim]
        
        # Expand anchor IDs to match batch dimension: [B, n_tables, n_anchor_pairs]
        anchor1_ids_expanded = anchor_pairs_a.unsqueeze(0).expand(batch_size, n_tables, n_anchor_pairs)  # [B, n_tables, n_anchor_pairs]
        anchor2_ids_expanded = anchor_pairs_b.unsqueeze(0).expand(batch_size, n_tables, n_anchor_pairs)  # [B, n_tables, n_anchor_pairs]
        
        # Gather anchor1 values: [B, n_tables, n_anchor_pairs]
        x_anchor1 = torch.gather(x_expanded, dim=2, index=anchor1_ids_expanded)
        
        # Gather anchor2 values: [B, n_tables, n_anchor_pairs]
        x_anchor2 = torch.gather(x_expanded, dim=2, index=anchor2_ids_expanded)
        
        # Compute deltas: [B, n_tables, n_anchor_pairs]
        deltas = x_anchor1 - x_anchor2
        
        # Form binary representation: [B, n_tables, n_anchor_pairs]
        bits = deltas.gt(cmp_eps).int()
        
        # Convert to integer lookup index: [B, n_tables]
        # lookup_index = sum(bits[i] << i) for each table
        lookup_indices = (bits << powers).sum(dim=2, dtype=torch.int32)  # [B, n_tables]

        # Find top K minimal absolute deltas for alternatives: [B, n_tables, n_alternatives]
        # Results are sorted in ascending order by absolute delta (smallest first)
        abs_deltas = deltas.abs()  # [B, n_tables, n_anchor_pairs]
        # Get top K smallest deltas and their indices
        min_delta_indices = torch.topk(
            abs_deltas, k=n_alternatives, dim=2, largest=False
        ).indices  # [B, n_tables, n_alternatives] each, sorted by ascending absolute delta
        
        # Create alternative indices by flipping bits at minimal delta positions
        # For each alternative, flip the bit at the corresponding anchor pair position
        # alt_index = lookup_index ^ (1 << min_delta_index)
        lookup_indices_expanded = lookup_indices.unsqueeze(2)  # [B, n_tables, 1] for broadcasting
        flip_masks = (1 << min_delta_indices).to(torch.int32)  # [B, n_tables, n_alternatives]
        lookup_alt_indices = (lookup_indices_expanded ^ flip_masks)  # [B, n_tables, n_alternatives]
        
        # Gather min deltas using min_delta_indices
        # min_delta_indices: [B, n_tables, n_alternatives] -> need to use as indices into deltas: [B, n_tables, n_anchor_pairs]
        # Use gather along the last dimension
        lookup_alt_deltas = torch.gather(deltas, dim=2, index=min_delta_indices)  # [B, n_tables, n_alternatives]

        # Gradient carriers (float tensors that mirror the indices)
        # Connect them to x so they're in the computation graph
        z = x.view(-1)[0] * 0
        lookup_indices_grad_c = z.expand(batch_size, n_tables)
        lookup_alt_indices_grad_c = z.expand(batch_size, n_tables, n_alternatives)
        
        # Save for backward
        # Compute min_deltas from deltas and min_delta_indices for uncertainty function
        min_deltas = torch.gather(deltas, dim=2, index=min_delta_indices)  # [B, n_tables, n_alternatives]
        ctx.save_for_backward(
            x,  # [B, input_dim] - needed for shape and device info
            anchor_pairs_a,  # [n_tables, n_anchor_pairs] - needed to get anchor IDs for gradient propagation
            anchor_pairs_b,  # [n_tables, n_anchor_pairs] - needed to get anchor IDs for gradient propagation
            min_delta_indices,  # [B, n_tables, n_alternatives] - indices of anchor pairs with minimal deltas
            min_deltas  # [B, n_tables, n_alternatives] - actual delta values for uncertainty function
        )
        ctx.cmp_eps = cmp_eps
        ctx.batch_size = batch_size
        ctx.n_tables = n_tables
        ctx.n_alternatives = n_alternatives
        
        return (
            lookup_indices,
            lookup_alt_indices,
            lookup_alt_deltas,
            lookup_indices_grad_c,
            lookup_alt_indices_grad_c
        )
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass.
        
        Propagates gradients through the anchor pairs using the uncertainty function.
        """
        (
            _,
            _,
            _,
            grad_lookup_indices_grad_c,
            grad_lookup_alt_indices_grad_c
        ) = grad_outputs
        
        (
            x, anchor_pairs_a, anchor_pairs_b, min_delta_indices, min_deltas
        ) = ctx.saved_tensors
        
        batch_size = ctx.batch_size
        n_tables = ctx.n_tables
        n_alternatives = ctx.n_alternatives
        device = x.device
        
        # Get gradients: [B, n_tables] and [B, n_tables, n_alternatives]
        grad_main = grad_lookup_indices_grad_c  # [B, n_tables]
        grad_alt = grad_lookup_alt_indices_grad_c  # [B, n_tables, n_alternatives]
        
        # Compute gradient difference for all alternatives: [B, n_tables, n_alternatives]
        grad_main_expanded = grad_main.unsqueeze(2)  # [B, n_tables, 1]
        grad_diff = grad_main_expanded - grad_alt  # [B, n_tables, n_alternatives]
        
        # Get min_deltas for all alternatives: [B, n_tables, n_alternatives]
        du = min_deltas.clone()  # [B, n_tables, n_alternatives]
        
        # Apply uncertainty function to all alternatives at once
        # if du > 0: du = 1/(1+|du|) * 0.5 * du
        # else: du = 1/(1+|du|) * -0.5 * du
        abs_du = du.abs()
        one_plus_abs = 1.0 + abs_du
        du_positive = du > 0
        du_negative = ~du_positive
        
        du[du_positive] = (1.0 / one_plus_abs[du_positive]) * 0.5 * du[du_positive]
        du[du_negative] = (1.0 / one_plus_abs[du_negative]) * (-0.5) * du[du_negative]
        
        # Multiply by gradient difference: [B, n_tables, n_alternatives]
        du = du * grad_diff
        
        # Get anchor IDs for all alternatives: [B, n_tables, n_alternatives]
        batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1)  # [B, 1, 1]
        table_indices = torch.arange(n_tables, device=device).view(1, n_tables, 1)  # [1, n_tables, 1]
        
        # Use advanced indexing to gather anchor IDs for all alternatives
        anchor1_ids = anchor_pairs_a[table_indices.squeeze(2), min_delta_indices]  # [B, n_tables, n_alternatives]
        anchor2_ids = anchor_pairs_b[table_indices.squeeze(2), min_delta_indices]  # [B, n_tables, n_alternatives]
        
        # Apply EPS check: [B, n_tables, n_alternatives]
        eps_mask = du.abs() > 1e-8
        
        # Initialize input gradients
        x_grad = torch.zeros_like(x)
        
        # Flatten all dimensions for efficient scatter
        # Expand batch and table indices to match alternatives dimension
        batch_expanded = batch_indices.expand(batch_size, n_tables, n_alternatives)  # [B, n_tables, n_alternatives]
        batch_flat = batch_expanded[eps_mask]  # [N]
        anchor1_flat = anchor1_ids[eps_mask]  # [N]
        anchor2_flat = anchor2_ids[eps_mask]  # [N]
        du_flat = du[eps_mask]  # [N]
        
        # Use scatter_add_ on flattened tensor for efficient accumulation
        x_grad_flat = x_grad.view(-1)  # [B * input_dim]
        indices1 = batch_flat * x.shape[1] + anchor1_flat  # [N]
        indices2 = batch_flat * x.shape[1] + anchor2_flat  # [N]
        x_grad_flat.scatter_add_(0, indices1, du_flat)
        x_grad_flat.scatter_add_(0, indices2, -du_flat)
        x_grad = x_grad_flat.view(batch_size, x.shape[1])
        
        return x_grad, None, None, None, None, None, None, None
