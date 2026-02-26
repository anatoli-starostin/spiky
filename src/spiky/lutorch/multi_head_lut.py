"""
Multi-head lookup table module combining AnchorPairsLookup and LProjection.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from anchor_pairs_lookup import AnchorPairsLookup
from l_projection import LProjection
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
    """
    
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        connected_pairs: bool = False,
        anchor_candidates: Optional[Union[torch.Tensor, Tuple[ChunkOfConnections, int]]] = None,
        cmp_eps: float = 0.0,
        random_seed: Optional[int] = None,
        n_alternatives: int = 1,
        smooth_mode: bool = False,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.n_alternatives = n_alternatives
        self.smooth_mode = smooth_mode
        
        # Total number of lookup tables
        n_lookup_tables = n_heads * tables_per_head
        
        # n_entries_per_table is derived from n_anchor_pairs
        n_entries_per_table = 2 ** n_anchor_pairs
        
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
            n_alternatives=n_alternatives
        )
        
        # Create LProjection: n_lookup_tables total
        self.projection = LProjection(
            n_lookup_tables=n_lookup_tables,
            n_entries_per_table=n_entries_per_table,
            n_outputs=n_outputs,
            n_alternatives=n_alternatives,
            smooth_mode=smooth_mode,
            device=device
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, input_dim]
            
        Returns:
            Output tensor of shape [B, n_heads, n_outputs]
        """
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
            lookup_indices, lookup_alt_indices = self.lookup(x, return_alternatives=return_alternatives)
            lookup_alt_deltas = None
            lookup_indices_grad_c = None
            lookup_alt_indices_grad_c = None
        
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

