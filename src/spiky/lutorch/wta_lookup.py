"""
Winner-Take-All (WTA) lookup implementation.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .abstract_lookup import AbstractLookup


class WTALookup(AbstractLookup):
    """
    Winner-Take-All lookup implementation.
    
    For each table, finds the anchor with maximum value and n_alternatives
    anchors closest to the maximum.
    
    Args:
        input_dim: Dimension of input tensor
        n_tables: Number of lookup tables
        n_anchors: Number of anchors per table
        n_alternatives: Number of alternative anchors (including the winner)
        anchors: Tensor of shape [n_tables, n_anchors] with input indices for each table
    """
    
    def __init__(
        self,
        input_dim: int,
        n_tables: int,
        n_anchors: int,
        n_alternatives: int,
        anchors: torch.Tensor
    ):
        if anchors.shape != (n_tables, n_anchors):
            raise ValueError(
                f"anchors must have shape [n_tables, n_anchors] = [{n_tables}, {n_anchors}], "
                f"got {anchors.shape}"
            )
        if n_alternatives > n_anchors:
            raise ValueError(
                f"n_alternatives ({n_alternatives}) must be <= n_anchors ({n_anchors})"
            )
        
        super().__init__(input_dim, n_tables, table_dim=n_anchors, n_alternatives=n_alternatives)
        
        self.n_anchors = n_anchors
        # Register anchors as buffer
        self.register_buffer('anchors', anchors)
    
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
                - lookup_indices: int [B, n_tables] - index of winner anchor
                - lookup_alt_indices: int [B, n_tables, n_alternatives] - indices of winner and alternatives
                - lookup_alt_deltas: float [B, n_tables, n_alternatives] - deltas from winner
                - lookup_indices_grad_c: float [B, n_tables]
                - lookup_alt_indices_grad_c: float [B, n_tables, n_alternatives]
            In eval mode:
                - lookup_indices: int [B, n_tables]
                - lookup_alt_indices: int [B, n_tables, n_alternatives]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get anchors (shape: [n_tables, n_anchors])
        anchors = self.anchors.to(device)
        
        if self.training:
            return self._forward_train(x, anchors, batch_size, device)
        else:
            return self._forward_eval(x, anchors, batch_size, device)
    
    def _forward_eval(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation forward pass."""
        # lookup_indices: [B, n_tables] - winner anchor index within table
        lookup_indices = torch.zeros(
            (batch_size, self.n_tables),
            dtype=torch.long,
            device=device
        )
        
        # lookup_alt_indices: [B, n_tables, n_alternatives]
        lookup_alt_indices = torch.zeros(
            (batch_size, self.n_tables, self.n_alternatives),
            dtype=torch.long,
            device=device
        )
        
        for table_idx in range(self.n_tables):
            # Get anchors for this table: [n_anchors]
            table_anchors = anchors[table_idx]  # [n_anchors]
            
            # Get input values at anchor positions: [B, n_anchors]
            x_anchors = x[:, table_anchors]  # [B, n_anchors]
            
            # Find winner (maximum value)
            winner_indices = x_anchors.argmax(dim=1)  # [B] - index within table
            
            # Find n_alternatives closest to winner
            # For each batch item, get the top n_alternatives values
            # We'll use the winner and the next n_alternatives-1 closest
            for b in range(batch_size):
                winner_idx = winner_indices[b].item()
                lookup_indices[b, table_idx] = winner_idx
                
                # Get values for this batch item
                values = x_anchors[b]  # [n_anchors]
                winner_value = values[winner_idx]
                
                # Compute distances from winner value
                distances = (values - winner_value).abs()  # [n_anchors]
                
                # Get top n_alternatives closest (including winner)
                # Sort by distance, then take first n_alternatives
                _, closest_indices = torch.topk(
                    distances, k=self.n_alternatives, largest=False
                )
                
                lookup_alt_indices[b, table_idx] = closest_indices
        
        return lookup_indices, lookup_alt_indices
    
    def _forward_train(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass with gradient carriers."""
        # Use autograd function for custom backward
        return WTALookupFunction.apply(
            x, anchors, batch_size, self.n_tables, self.n_anchors,
            self.n_alternatives
        )


class WTALookupFunction(torch.autograd.Function):
    """Custom autograd function for WTA lookup with gradient propagation."""
    
    @staticmethod
    def forward(ctx, *args):
        """
        Forward pass.
        
        Args:
            ctx: Context object
            *args: x, anchors, batch_size, n_tables, n_anchors, n_alternatives
        
        Returns:
            lookup_indices: int [B, n_tables]
            lookup_alt_indices: int [B, n_tables, n_alternatives]
            lookup_alt_deltas: float [B, n_tables, n_alternatives]
            lookup_indices_grad_c: float [B, n_tables]
            lookup_alt_indices_grad_c: float [B, n_tables, n_alternatives]
        """
        x, anchors, batch_size, n_tables, n_anchors, n_alternatives = args
        device = x.device
        # lookup_indices: [B, n_tables] - winner anchor index
        lookup_indices = torch.zeros(
            (batch_size, n_tables),
            dtype=torch.long,
            device=device
        )
        
        # lookup_alt_indices: [B, n_tables, n_alternatives]
        lookup_alt_indices = torch.zeros(
            (batch_size, n_tables, n_alternatives),
            dtype=torch.long,
            device=device
        )
        
        # lookup_alt_deltas: [B, n_tables, n_alternatives] - deltas from winner
        lookup_alt_deltas = torch.zeros(
            (batch_size, n_tables, n_alternatives),
            dtype=x.dtype,
            device=device
        )
        
        # Gradient carriers
        lookup_indices_grad_c = torch.zeros(
            (batch_size, n_tables),
            dtype=x.dtype,
            device=device,
            requires_grad=True
        )
        
        lookup_alt_indices_grad_c = torch.zeros(
            (batch_size, n_tables, n_alternatives),
            dtype=x.dtype,
            device=device,
            requires_grad=True
        )
        
        # Store winner and prewinner (second best) for backward
        winner_values = torch.zeros(
            (batch_size, n_tables),
            dtype=x.dtype,
            device=device
        )
        prewinner_values = torch.zeros(
            (batch_size, n_tables),
            dtype=x.dtype,
            device=device
        )
        winner_indices = torch.zeros(
            (batch_size, n_tables),
            dtype=torch.long,
            device=device
        )
        prewinner_indices = torch.zeros(
            (batch_size, n_tables),
            dtype=torch.long,
            device=device
        )
        
        for table_idx in range(n_tables):
            # Get anchors for this table: [n_anchors]
            table_anchors = anchors[table_idx]  # [n_anchors]
            
            # Get input values at anchor positions: [B, n_anchors]
            x_anchors = x[:, table_anchors]  # [B, n_anchors]
            
            # Find winner (maximum value)
            winner_idx = x_anchors.argmax(dim=1)  # [B] - index within table
            winner_indices[:, table_idx] = winner_idx
            
            # Get winner values
            for b in range(batch_size):
                winner_values[b, table_idx] = x_anchors[b, winner_idx[b]]
            
            # Find prewinner (second maximum)
            # Create a mask to exclude winner
            for b in range(batch_size):
                w_idx = winner_idx[b].item()
                # Set winner value to -inf temporarily
                x_anchors_masked = x_anchors[b].clone()
                x_anchors_masked[w_idx] = float('-inf')
                prewinner_idx = x_anchors_masked.argmax().item()
                prewinner_indices[b, table_idx] = prewinner_idx
                prewinner_values[b, table_idx] = x_anchors[b, prewinner_idx]
            
            # Store lookup indices
            lookup_indices[:, table_idx] = winner_idx
            
            # Find n_alternatives closest to winner
            for b in range(batch_size):
                values = x_anchors[b]  # [n_anchors]
                winner_value = values[winner_idx[b]]
                
                # Compute distances from winner value
                distances = (values - winner_value).abs()  # [n_anchors]
                
                # Get top n_alternatives closest (including winner)
                _, closest_indices = torch.topk(
                    distances, k=n_alternatives, largest=False
                )
                
                lookup_alt_indices[b, table_idx] = closest_indices
                
                # Compute deltas (differences from winner)
                alt_values = values[closest_indices]
                lookup_alt_deltas[b, table_idx] = alt_values - winner_value
        
        # Initialize gradient carriers as float versions of indices
        # Connect them to x so they're in the computation graph
        lookup_indices_grad_c = lookup_indices.float() + 0.0 * x.sum() * 0.0
        lookup_alt_indices_grad_c = lookup_alt_indices.float() + 0.0 * x.sum() * 0.0
        
        # Save for backward
        ctx.save_for_backward(
            x, anchors, winner_indices, prewinner_indices,
            winner_values, prewinner_values,
            lookup_indices_grad_c, lookup_alt_indices_grad_c
        )
        ctx.batch_size = batch_size
        ctx.n_tables = n_tables
        ctx.n_anchors = n_anchors
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
        
        Propagates gradients through winner/prewinner pairs using the uncertainty function.
        Similar to propagate_through_detectors_logic in ANDN.
        """
        (
            grad_lookup_indices,
            grad_lookup_alt_indices,
            grad_lookup_alt_deltas,
            grad_lookup_indices_grad_c,
            grad_lookup_alt_indices_grad_c
        ) = grad_outputs
        
        (
            x, anchors, winner_indices, prewinner_indices,
            winner_values, prewinner_values,
            lookup_indices_grad_c, lookup_alt_indices_grad_c
        ) = ctx.saved_tensors
        
        batch_size = ctx.batch_size
        n_tables = ctx.n_tables
        device = x.device
        
        # Initialize input gradients
        x_grad = torch.zeros_like(x)
        
        # Process each table
        for table_idx in range(n_tables):
            table_anchors = anchors[table_idx]  # [n_anchors]
            
            # Get gradients for this table
            # grad_lookup_indices_grad_c: [B, n_tables] -> [B]
            table_grad_main = grad_lookup_indices_grad_c[:, table_idx]  # [B]
            
            # grad_lookup_alt_indices_grad_c: [B, n_tables, n_alternatives]
            # We need to aggregate alternatives - use the first one (winner) as reference
            table_grad_alt = grad_lookup_alt_indices_grad_c[:, table_idx, 0]  # [B]
            
            # Get winner and prewinner indices and values
            winner_idx = winner_indices[:, table_idx]  # [B]
            prewinner_idx = prewinner_indices[:, table_idx]  # [B]
            winner_val = winner_values[:, table_idx]  # [B]
            prewinner_val = prewinner_values[:, table_idx]  # [B]
            
            # Compute delta (difference between winner and prewinner)
            du = winner_val - prewinner_val  # [B]
            
            # Apply uncertainty function (same as in propagate_through_detectors_logic)
            # if du > 0: du = 1/(1+|du|) * 0.5 * du
            # else: du = 1/(1+|du|) * -0.5 * du
            abs_du = du.abs()
            one_plus_abs = 1.0 + abs_du
            du_positive = du > 0
            du_negative = ~du_positive
            
            du[du_positive] = (1.0 / one_plus_abs[du_positive]) * 0.5 * du[du_positive]
            du[du_negative] = (1.0 / one_plus_abs[du_negative]) * (-0.5) * du[du_negative]
            
            # Multiply by gradient difference
            grad_diff = table_grad_main - table_grad_alt  # [B]
            du = du * grad_diff  # [B]
            
            # Propagate to input through winner and prewinner anchors
            for b in range(batch_size):
                if abs(du[b].item()) > 1e-8:  # EPS check
                    w_idx = winner_idx[b].item()
                    pw_idx = prewinner_idx[b].item()
                    
                    # Get actual input indices
                    winner_input_idx = table_anchors[w_idx].item()
                    prewinner_input_idx = table_anchors[pw_idx].item()
                    
                    du_val = du[b].item()
                    x_grad[b, winner_input_idx] += du_val
                    x_grad[b, prewinner_input_idx] -= du_val
        
        return x_grad, None, None, None, None, None

