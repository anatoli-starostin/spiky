"""
Multi-head lookup table module combining AnchorPairsLookup and LProjection.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from spiky.lutorch.anchor_pairs_lookup import AnchorPairsLookup
from spiky.lutorch.l_projection import LProjection
from spiky.lutorch.lut_helpers import UncertaintyMode
from spiky.util.chunk_of_connections import ChunkOfConnections


class MultiHeadLut(nn.Module):
    """
    Multi-head lookup table module.
    
    Combines AnchorPairsLookup and LProjection to create a multi-head
    structure using lookup tables.
    
    Args:
        input_dim: Dimension of input tensor
        n_heads: Number of heads
        n_outputs: Number of output dimensions per head
        n_anchor_pairs: Number of anchor pairs per table (determines n_entries_per_table = 2**n_anchor_pairs)
        tables_per_head: Number of lookup tables per head (default: 1)
        n_buckets: Number of buckets for bucketized lookup (default: 1). If > 1, forward expects bucket_indices input.
        connected_pairs: If True, anchor pairs form a connected graph (default: False)
        anchor_candidates: Optional. Either:
                          - torch.Tensor: Shape [n_heads, tables_per_head, max_anchors_per_table] with input indices
                          - Tuple[ChunkOfConnections, int]: ChunkOfConnections with custom ids_shift
                          - None: Uses all input indices (default)
        cmp_eps: Epsilon for comparison (default: 0.0)
        random_seed: Random seed for anchor pair sampling
        n_alternatives: Number of alternative lookup indices per table (default: 1)
        smooth_mode: If True, use smooth interpolation in LProjection (default: False)
        device: Device to place buffers on
        anchor_initialization: "default" or "balanced". "balanced" matches spike_QK (randperm-based even coverage).
    """
    
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        n_buckets: int = 1,
        connected_pairs: bool = False,
        anchor_candidates: Optional[Union[torch.Tensor, Tuple[ChunkOfConnections, int]]] = None,
        cmp_eps: float = 0.0,
        random_seed: Optional[int] = None,
        n_alternatives: int = 1,
        smooth_mode: bool = False,
        device: Optional[torch.device] = None,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
        anchor_initialization: str = "default",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.n_buckets = n_buckets
        self.n_alternatives = n_alternatives
        self.smooth_mode = smooth_mode
        self.uncertainty_mode = uncertainty_mode
        
        # Total number of lookup tables
        n_lookup_tables = n_heads * tables_per_head
        
        # n_entries_per_table is derived from n_anchor_pairs, multiplied by n_buckets
        n_entries_per_table = (2 ** n_anchor_pairs) * n_buckets
        
        # Reshape anchor_candidates if it's a tensor: [n_heads, tables_per_head, max_anchors_per_table] -> [n_heads * tables_per_head, max_anchors_per_table]
        reshaped_anchor_candidates = anchor_candidates
        if isinstance(anchor_candidates, torch.Tensor):
            reshaped_anchor_candidates = anchor_candidates.view(n_lookup_tables, -1)
        
        # Create AnchorPairsLookup: n_lookup_tables total
        self.lookup = AnchorPairsLookup(
            input_dim=input_dim,
            n_tables=n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            connected_pairs=connected_pairs,
            anchor_candidates=reshaped_anchor_candidates,
            cmp_eps=cmp_eps,
            random_seed=random_seed,
            device=device,
            n_alternatives=n_alternatives,
            uncertainty_mode=uncertainty_mode,
            anchor_initialization=anchor_initialization,
        )
        
        # Create LProjection: n_lookup_tables total
        self.projection = LProjection(
            n_lookup_tables=n_lookup_tables,
            n_entries_per_table=n_entries_per_table,
            n_outputs=n_outputs,
            n_alternatives=n_alternatives,
            smooth_mode=smooth_mode,
            device=device,
            uncertainty_mode=uncertainty_mode,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        bucket_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, input_dim]
            bucket_indices: Optional integer tensor of shape [B] with bucket indices for each batch element.
                           Required if n_buckets > 1.
            
        Returns:
            Output tensor of shape [B, n_heads, n_outputs]
        """
        # Validate bucket_indices if n_buckets > 1
        if self.n_buckets > 1:
            if bucket_indices is None:
                raise ValueError(f"bucket_indices is required when n_buckets={self.n_buckets} > 1")
            assert bucket_indices.shape == (x.shape[0],), \
                f"bucket_indices must have shape [{x.shape[0]}], got {bucket_indices.shape}, expected [{x.shape[0]}]"
            assert bucket_indices.dtype == torch.int32 or bucket_indices.dtype == torch.int64, \
                f"bucket_indices must be integer tensor, got {bucket_indices.dtype}"

        # Determine return_alternatives: always True in training, only True in eval if smooth_mode
        return_alternatives = self.training or self.smooth_mode
        
        # Get lookup indices from AnchorPairsLookup
        if self.training:
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                lookup_indices_grad_c,
                lookup_alt_indices_grad_c
            ) = self.lookup(x, return_alternatives=return_alternatives)
        else:
            lookup_indices, lookup_alt_indices, lookup_alt_deltas = self.lookup(x, return_alternatives=return_alternatives)
            lookup_indices_grad_c = None
            lookup_alt_indices_grad_c = None
        
        # Apply bucket modification if n_buckets > 1
        if self.n_buckets > 1:
            # Ensure bucket_indices has the right dtype and shape for broadcasting
            bucket_indices = bucket_indices.to(lookup_indices.dtype)  # [B]
            bucket_indices_expanded = bucket_indices.unsqueeze(1)  # [B, 1]
            
            # Modify lookup_indices: lookup_indices = lookup_indices * n_buckets + bucket_indices
            lookup_indices = lookup_indices * self.n_buckets + bucket_indices_expanded  # [B, n_tables]
            
            # Modify lookup_alt_indices similarly
            if lookup_alt_indices is not None:
                bucket_indices_alt = bucket_indices.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                lookup_alt_indices = lookup_alt_indices * self.n_buckets + bucket_indices_alt  # [B, n_tables, n_alternatives]
        
        # Project through LProjection
        # LProjection returns [B, n_lookup_tables, n_outputs]
        output = self.projection(
            lookup_indices=lookup_indices,
            lookup_alt_indices=lookup_alt_indices,
            lookup_alt_deltas=lookup_alt_deltas,
            lookup_indices_grad_c=lookup_indices_grad_c,
            lookup_alt_indices_grad_c=lookup_alt_indices_grad_c
        )
        
        # Reshape and sum: [B, n_lookup_tables, n_outputs] -> [B, n_heads, tables_per_head, n_outputs] -> [B, n_heads, n_outputs]
        # output: [B, n_heads * tables_per_head, n_outputs]
        output = output.view(-1, self.n_heads, self.tables_per_head, self.n_outputs)
        # Sum over tables_per_head dimension
        output = output.sum(dim=2)  # [B, n_heads, n_outputs]
        
        return output
