"""
Abstract base class for lookup operations.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class AbstractLookup(nn.Module):
    """
    Abstract base class for lookup operations.
    
    Args:
        input_dim: Dimension of input tensor
        n_tables: Number of lookup tables
        table_dim: Dimension of each lookup table (number of entries)
        n_alternatives: Number of alternative lookup indices per table
    """
    
    def __init__(
        self,
        input_dim: int,
        n_tables: int,
        table_dim: int,
        n_alternatives: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_tables = n_tables
        self.table_dim = table_dim
        self.n_alternatives = n_alternatives
    
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
                - lookup_alt_indices: int [B, n_tables, n_alternatives]
                - lookup_alt_deltas: float [B, n_tables, n_alternatives]
                - lookup_indices_grad_c: float [B, n_tables]
                - lookup_alt_indices_grad_c: float [B, n_tables, n_alternatives]
            In eval mode:
                - lookup_indices: int [B, n_tables]
                - lookup_alt_indices: int [B, n_tables, n_alternatives]
        """
        raise NotImplementedError("Subclasses must implement forward method")

