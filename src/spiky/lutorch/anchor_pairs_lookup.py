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
        device: Optional[torch.device] = None
    ):
        table_dim = 2 ** n_anchor_pairs
        super().__init__(input_dim, n_tables, table_dim, n_alternatives=1)
        
        self.n_anchor_pairs = n_anchor_pairs
        self.connected_pairs = connected_pairs
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
        
        # Get anchor pairs from sampler and register as buffer
        anchor_pairs = anchor_sampler.get_anchor_pairs()
        # AnchorSampler returns [n_tables, n_anchor_pairs, 2], which is what we need
        self.register_buffer('anchor_pairs', anchor_pairs)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, input_dim]
            
        Returns:
            In training mode:
                - lookup_indices: int [B, n_tables]
                - lookup_alt_indices: int [B, n_tables, 1]
                - lookup_alt_deltas: float [B, n_tables, 1]
                - lookup_indices_grad_c: float [B, n_tables]
                - lookup_alt_indices_grad_c: float [B, n_tables, 1]
            In eval mode:
                - lookup_indices: int [B, n_tables]
                - lookup_alt_indices: int [B, n_tables, 1]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get anchor pairs (shape: [n_tables, n_anchor_pairs, 2])
        anchor_pairs = self.anchor_pairs.to(device)
        
        if self.training:
            return self._forward_train(x, anchor_pairs, batch_size, device)
        else:
            return self._forward_eval(x, anchor_pairs, batch_size, device)
    
    def _forward_eval(
        self,
        x: torch.Tensor,
        anchor_pairs: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation forward pass."""
        # lookup_indices: [B, n_tables]
        lookup_indices = torch.zeros(
            (batch_size, self.n_tables),
            dtype=torch.long,
            device=device
        )
        
        # lookup_alt_indices: [B, n_tables, 1] (same as lookup_indices for anchor pairs)
        lookup_alt_indices = torch.zeros(
            (batch_size, self.n_tables, 1),
            dtype=torch.long,
            device=device
        )
        
        for table_idx in range(self.n_tables):
            # Get anchor pairs for this table: [n_anchor_pairs, 2]
            table_pairs = anchor_pairs[table_idx]  # [n_anchor_pairs, 2]
            
            # Compute deltas: [B, n_anchor_pairs]
            anchor1_ids = table_pairs[:, 0]  # [n_anchor_pairs]
            anchor2_ids = table_pairs[:, 1]  # [n_anchor_pairs]
            
            x_anchor1 = x[:, anchor1_ids]  # [B, n_anchor_pairs]
            x_anchor2 = x[:, anchor2_ids]  # [B, n_anchor_pairs]
            deltas = x_anchor1 - x_anchor2  # [B, n_anchor_pairs]
            
            # Form binary representation
            # Compare with epsilon
            bits = (deltas > self.cmp_eps).long()  # [B, n_anchor_pairs]
            
            # Convert to integer lookup index
            # lookup_index = sum(bits[i] * 2^i)
            powers = torch.arange(
                self.n_anchor_pairs,
                device=device,
                dtype=torch.long
            )
            lookup_indices[:, table_idx] = (bits * (2 ** powers)).sum(dim=1)
        
        # For anchor pairs, alt_indices is the same as lookup_indices
        lookup_alt_indices[:, :, 0] = lookup_indices
        
        return lookup_indices, lookup_alt_indices
    
    def _forward_train(
        self,
        x: torch.Tensor,
        anchor_pairs: torch.Tensor,
        batch_size: int,
        _: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass with gradient carriers."""
        # Use autograd function for custom backward
        return AnchorPairsLookupFunction.apply(
            x, anchor_pairs, self.n_anchor_pairs, self.cmp_eps,
            batch_size, self.n_tables
        )


class AnchorPairsLookupFunction(torch.autograd.Function):
    """Custom autograd function for anchor pairs lookup with gradient propagation."""
    
    @staticmethod
    def forward(ctx, *args):
        """
        Forward pass.
        
        Args:
            ctx: Context object
            *args: x, anchor_pairs, n_anchor_pairs, cmp_eps, batch_size, n_tables
        
        Returns:
            lookup_indices: int [B, n_tables]
            lookup_alt_indices: int [B, n_tables, 1]
            lookup_alt_deltas: float [B, n_tables, 1]
            lookup_indices_grad_c: float [B, n_tables]
            lookup_alt_indices_grad_c: float [B, n_tables, 1]
        """
        x, anchor_pairs, n_anchor_pairs, cmp_eps, batch_size, n_tables = args
        device = x.device
        # anchor_pairs: [n_tables, n_anchor_pairs, 2]
        # Extract anchor IDs: [n_tables, n_anchor_pairs]
        anchor1_ids = anchor_pairs[:, :, 0]  # [n_tables, n_anchor_pairs]
        anchor2_ids = anchor_pairs[:, :, 1]  # [n_tables, n_anchor_pairs]
        
        # Use advanced indexing to get anchor values for all tables at once
        # x: [B, input_dim]
        # We need: [B, n_tables, n_anchor_pairs]
        # Expand x: [B, 1, input_dim] then expand to [B, n_tables, input_dim]
        x_expanded = x.unsqueeze(1).expand(batch_size, n_tables, x.shape[1])  # [B, n_tables, input_dim]
        
        # Expand anchor IDs to match batch dimension: [B, n_tables, n_anchor_pairs]
        anchor1_ids_expanded = anchor1_ids.unsqueeze(0).expand(batch_size, n_tables, n_anchor_pairs)  # [B, n_tables, n_anchor_pairs]
        anchor2_ids_expanded = anchor2_ids.unsqueeze(0).expand(batch_size, n_tables, n_anchor_pairs)  # [B, n_tables, n_anchor_pairs]
        
        # Gather anchor1 values: [B, n_tables, n_anchor_pairs]
        x_anchor1 = torch.gather(x_expanded, dim=2, index=anchor1_ids_expanded)
        
        # Gather anchor2 values: [B, n_tables, n_anchor_pairs]
        x_anchor2 = torch.gather(x_expanded, dim=2, index=anchor2_ids_expanded)
        
        # Compute deltas: [B, n_tables, n_anchor_pairs]
        deltas = x_anchor1 - x_anchor2
        
        # Form binary representation: [B, n_tables, n_anchor_pairs]
        bits = (deltas > cmp_eps).long()
        
        # Convert to integer lookup index: [B, n_tables]
        # lookup_index = sum(bits[i] * 2^i) for each table
        powers = torch.arange(n_anchor_pairs, device=device, dtype=torch.long)
        powers_expanded = powers.view(1, 1, n_anchor_pairs)  # [1, 1, n_anchor_pairs]
        lookup_indices = (bits * (2 ** powers_expanded)).sum(dim=2)  # [B, n_tables]
        
        # Find anchor pair with minimum absolute delta: [B, n_tables]
        abs_deltas = deltas.abs()  # [B, n_tables, n_anchor_pairs]
        min_delta_indices = abs_deltas.argmin(dim=2)  # [B, n_tables]
        
        # Gather min deltas using min_delta_indices
        # min_delta_indices: [B, n_tables] -> need to use as indices into deltas: [B, n_tables, n_anchor_pairs]
        # Use gather along the last dimension
        min_delta_indices_expanded = min_delta_indices.unsqueeze(2)  # [B, n_tables, 1]
        min_deltas = torch.gather(deltas, dim=2, index=min_delta_indices_expanded).squeeze(2)  # [B, n_tables]
        
        # lookup_alt_indices: [B, n_tables, 1] - same as lookup_indices for anchor pairs
        lookup_alt_indices = lookup_indices.unsqueeze(2)  # [B, n_tables, 1]
        
        # lookup_alt_deltas: [B, n_tables, 1]
        lookup_alt_deltas = min_deltas.unsqueeze(2)  # [B, n_tables, 1]

        # Gradient carriers (float tensors that mirror the indices)
        # Connect them to x so they're in the computation graph
        z = x.view(-1)[0] * 0
        lookup_indices_grad_c = z.expand(batch_size, n_tables)
        lookup_alt_indices_grad_c = z.expand(batch_size, n_tables, 1)
        
        # Save for backward
        ctx.save_for_backward(
            x, anchor_pairs, min_deltas, min_delta_indices,
            lookup_indices_grad_c, lookup_alt_indices_grad_c
        )
        ctx.n_anchor_pairs = n_anchor_pairs
        ctx.cmp_eps = cmp_eps
        ctx.batch_size = batch_size
        ctx.n_tables = n_tables
        
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
            grad_lookup_indices,
            grad_lookup_alt_indices,
            grad_lookup_alt_deltas,
            grad_lookup_indices_grad_c,
            grad_lookup_alt_indices_grad_c
        ) = grad_outputs
        
        (
            x, anchor_pairs, min_deltas, min_delta_indices,
            lookup_indices_grad_c, lookup_alt_indices_grad_c
        ) = ctx.saved_tensors
        
        batch_size = ctx.batch_size
        n_tables = ctx.n_tables
        device = x.device
        
        # Get gradients for all tables: [B, n_tables]
        grad_main = grad_lookup_indices_grad_c  # [B, n_tables]
        grad_alt = grad_lookup_alt_indices_grad_c.squeeze(2)  # [B, n_tables]
        
        # Compute gradient difference: [B, n_tables]
        grad_diff = grad_main - grad_alt
        
        # Compute uncertainty function gradient (du): [B, n_tables]
        # du = min_delta / (1 + |min_delta|)^2 * grad_diff
        # But with sign handling like in propagate_through_detector
        du = min_deltas.clone()  # [B, n_tables]
        
        # Apply uncertainty function
        # if du > 0: du = 1/(1+|du|) * 0.5 * du
        # else: du = 1/(1+|du|) * -0.5 * du
        abs_du = du.abs()
        one_plus_abs = 1.0 + abs_du
        du_positive = du > 0
        du_negative = ~du_positive
        
        du[du_positive] = (1.0 / one_plus_abs[du_positive]) * 0.5 * du[du_positive]
        du[du_negative] = (1.0 / one_plus_abs[du_negative]) * (-0.5) * du[du_negative]
        
        # Multiply by gradient difference: [B, n_tables]
        du = du * grad_diff
        
        # Get anchor IDs for minimum delta pairs: [B, n_tables]
        # min_delta_indices: [B, n_tables] - indices into anchor_pairs[:, :, 0/1]
        # We need to gather the anchor IDs using these indices
        batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, n_tables)
        table_indices = torch.arange(n_tables, device=device).view(1, n_tables).expand(batch_size, n_tables)
        
        # Gather anchor1 and anchor2 IDs: [B, n_tables]
        anchor1_ids = anchor_pairs[table_indices, min_delta_indices, 0]  # [B, n_tables]
        anchor2_ids = anchor_pairs[table_indices, min_delta_indices, 1]  # [B, n_tables]
        
        # Apply EPS check: [B, n_tables]
        eps_mask = du.abs() > 1e-8
        
        # Initialize input gradients
        x_grad = torch.zeros_like(x)
        
        # Scatter gradients to input
        # For each (batch, table) pair, add du to anchor1 and subtract from anchor2
        # Flatten indices and values for efficient accumulation
        batch_flat = batch_indices[eps_mask]  # [N]
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
        
        return x_grad, None, None, None, None, None
