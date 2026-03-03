"""
LProjection module for lookup table-based projection.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LProjection(nn.Module):
    """
    Lookup table projection module.
    
    Takes lookup indices and performs weighted lookups from internal weight tables.
    Supports basic (non-smooth) and smooth modes.
    
    Args:
        n_lookup_tables: Number of lookup tables
        n_entries_per_table: Number of entries per table
        n_alternatives: Number of alternative indices per table (default: 1)
        n_outputs: Number of output dimensions
        smooth_mode: If True, use smooth interpolation with uncertainty function (default: False)
    """
    
    def __init__(
        self,
        n_lookup_tables: int,
        n_entries_per_table: int,
        n_outputs: int,
        n_alternatives: int = 1,
        smooth_mode: bool = False,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.n_lookup_tables = n_lookup_tables
        self.n_entries_per_table = n_entries_per_table
        self.n_alternatives = n_alternatives
        self.n_outputs = n_outputs
        self.smooth_mode = smooth_mode
        
        # Initialize weight tensor: [n_lookup_tables, n_entries_per_table, n_outputs]
        self.weights = nn.Parameter(torch.zeros(n_lookup_tables, n_entries_per_table, n_outputs, device=device))
        
        # Cache for expanded table indices (recalculated when batch_size changes)
        self._cached_table_indices_expanded = None
        self._cached_table_indices_flat = None
        self._cached_table_indices_expanded_alt = None
        self._cached_table_indices_alt_flat = None
    
    def forward(
        self,
        lookup_indices: torch.Tensor,
        lookup_alt_indices: Optional[torch.Tensor] = None,
        lookup_alt_deltas: Optional[torch.Tensor] = None,
        lookup_indices_grad_c: Optional[torch.Tensor] = None,
        lookup_alt_indices_grad_c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            lookup_indices: int64 [B, n_lookup_tables]
            lookup_alt_indices: int64 [B, n_lookup_tables, n_alternatives] (may be None)
            lookup_alt_deltas: float [B, n_lookup_tables, n_alternatives] (may be None)
            lookup_indices_grad_c: float [B, n_lookup_tables] (may be None in eval)
            lookup_alt_indices_grad_c: float [B, n_lookup_tables, n_alternatives] (may be None in eval)
        
        Returns:
            Output tensor of shape [B, n_lookup_tables, n_outputs]
        """
        if self.training:
            return self._forward_train(
                lookup_indices, lookup_alt_indices, lookup_alt_deltas,
                lookup_indices_grad_c, lookup_alt_indices_grad_c
            )[0]
        else:
            return self._forward_eval(
                lookup_indices, lookup_alt_indices, lookup_alt_deltas
            )
    
    def _prepare_table_indices(self, batch_size: int, device: torch.device) -> None:
        """
        Prepare expanded table indices (cached for efficiency).
        
        Args:
            batch_size: Current batch size
            device: Device to place tensors on
        """
        if (
            self._cached_table_indices_expanded is None
            or self._cached_table_indices_expanded.shape[0] != batch_size
            or self._cached_table_indices_expanded.device != device
        ):
            table_indices = torch.arange(self.n_lookup_tables, dtype=torch.long, device=device).view(1, self.n_lookup_tables)
            self._cached_table_indices_expanded = table_indices.expand(batch_size, self.n_lookup_tables)
            self._cached_table_indices_flat = self._cached_table_indices_expanded.flatten()  # [batch_size * n_lookup_tables]
            
            # Precompute alternative indices: [batch_size, n_lookup_tables, n_alternatives]
            self._cached_table_indices_expanded_alt = self._cached_table_indices_expanded.unsqueeze(2).expand(batch_size, self.n_lookup_tables, self.n_alternatives)
            self._cached_table_indices_alt_flat = self._cached_table_indices_expanded_alt.flatten()  # [batch_size * n_lookup_tables * n_alternatives]
    
    def _forward_eval(
        self,
        lookup_indices: torch.Tensor,
        lookup_alt_indices: Optional[torch.Tensor],
        lookup_alt_deltas: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Evaluation forward pass."""
        batch_size = lookup_indices.shape[0]
        # Prepare expanded table indices (cached for efficiency)
        self._prepare_table_indices(batch_size, lookup_indices.device)

        if not self.smooth_mode:
            # Non-smooth: just lookup weights by lookup_indices
            # lookup_indices: [B, n_lookup_tables]
            # weights: [n_lookup_tables, n_entries_per_table, n_outputs]
            # Output: [B, n_lookup_tables, n_outputs]
            # Gather weights: [B, n_lookup_tables, n_outputs]
            output = self.weights[self._cached_table_indices_expanded, lookup_indices]
            return output
        else:
            # Smooth mode: weighted combination using uncertainty function
            assert lookup_alt_indices is not None, "lookup_alt_indices required in smooth mode"
            assert lookup_alt_deltas is not None, "lookup_alt_deltas required in smooth mode"
            
            # Apply uncertainty function: U(delta) = 0.5 / (1 + |delta|)
            abs_deltas = lookup_alt_deltas.abs()
            uncertainty = 0.5 / (1.0 + abs_deltas)  # [B, n_lookup_tables, n_alternatives]
            
            # Weight for main indices: sum(1 - U(delta)) / n_alternatives
            main_weight = (1.0 - uncertainty).sum(dim=2) / self.n_alternatives  # [B, n_lookup_tables]
            
            # Weight for alt indices: U(delta) / n_alternatives
            alt_weight = uncertainty / self.n_alternatives  # [B, n_lookup_tables, n_alternatives]
            
            # Lookup main weights using pre-computed table indices
            main_weights = self.weights[self._cached_table_indices_expanded, lookup_indices]  # [B, n_lookup_tables, n_outputs]
            
            # Lookup alt weights using pre-computed alternative indices
            alt_weights = self.weights[self._cached_table_indices_expanded_alt, lookup_alt_indices]  # [B, n_lookup_tables, n_alternatives, n_outputs]
            
            # Weighted combination
            # main_weights: [B, n_lookup_tables, n_outputs]
            # main_weight: [B, n_lookup_tables] -> [B, n_lookup_tables, 1]
            output = main_weights * main_weight.unsqueeze(2)  # [B, n_lookup_tables, n_outputs]
            
            # alt_weights: [B, n_lookup_tables, n_alternatives, n_outputs]
            # alt_weight: [B, n_lookup_tables, n_alternatives] -> [B, n_lookup_tables, n_alternatives, 1]
            alt_weighted = alt_weights * alt_weight.unsqueeze(3)  # [B, n_lookup_tables, n_alternatives, n_outputs]
            output = output + alt_weighted.sum(dim=2)  # [B, n_lookup_tables, n_outputs]
            
            return output
    
    def _forward_train(
        self,
        lookup_indices: torch.Tensor,
        lookup_alt_indices: Optional[torch.Tensor],
        lookup_alt_deltas: Optional[torch.Tensor],
        lookup_indices_grad_c: Optional[torch.Tensor],
        lookup_alt_indices_grad_c: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Training forward pass with autograd function."""
        batch_size = lookup_indices.shape[0]
        
        # Prepare expanded table indices (cached for efficiency)
        self._prepare_table_indices(batch_size, lookup_indices.device)
        
        return LProjectionFunction.apply(
            self.weights, lookup_indices, lookup_alt_indices, lookup_alt_deltas,
            lookup_indices_grad_c, lookup_alt_indices_grad_c,
            self.smooth_mode, self.n_alternatives, self._cached_table_indices_expanded, self._cached_table_indices_flat,
            self._cached_table_indices_expanded_alt, self._cached_table_indices_alt_flat
        )


class LProjectionFunction(torch.autograd.Function):
    """Custom autograd function for LProjection with gradient propagation."""
    
    @staticmethod
    def forward(ctx, *args):
        """
        Forward pass.
        
        Args:
            ctx: Context object
            *args: weights, lookup_indices, lookup_alt_indices, lookup_alt_deltas,
                   lookup_indices_grad_c, lookup_alt_indices_grad_c, smooth_mode, n_alternatives, table_indices_expanded, table_indices_flat
        
        Returns:
            Output tensor of shape [B, n_lookup_tables, n_outputs]
        """
        (
            weights, lookup_indices, lookup_alt_indices, lookup_alt_deltas,
            lookup_indices_grad_c, lookup_alt_indices_grad_c, smooth_mode,
            n_alternatives, table_indices_expanded, table_indices_flat,
            table_indices_expanded_alt, table_indices_alt_flat
        ) = args
        
        batch_size = lookup_indices.shape[0]
        n_lookup_tables = lookup_indices.shape[1]

        if not smooth_mode:
            # Non-smooth: just lookup weights using pre-computed table indices
            output = weights[table_indices_expanded, lookup_indices]  # [B, n_lookup_tables, n_outputs]
            
            # Save for backward (include alt indices and gradient carriers if provided for gradient computation)
            ctx.save_for_backward(weights, lookup_indices, lookup_alt_indices, lookup_indices_grad_c, lookup_alt_indices_grad_c, table_indices_flat)
            ctx.smooth_mode = False
            ctx.batch_size = batch_size
            ctx.n_lookup_tables = n_lookup_tables
        else:
            # Smooth mode: weighted combination
            assert lookup_alt_indices is not None, "lookup_alt_indices required in smooth mode"
            assert lookup_alt_deltas is not None, "lookup_alt_deltas required in smooth mode"
            
            # Apply uncertainty function: U(delta) = 0.5 / (1 + |delta|)
            abs_deltas = lookup_alt_deltas.abs()
            uncertainty = 0.5 / (1.0 + abs_deltas)  # [B, n_lookup_tables, n_alternatives]
            
            # Weight for main indices: sum(1 - U(delta)) / n_alternatives
            main_weight = (1.0 - uncertainty).sum(dim=2) / n_alternatives  # [B, n_lookup_tables]
            
            # Weight for alt indices: U(delta) / n_alternatives
            alt_weight = uncertainty / n_alternatives  # [B, n_lookup_tables, n_alternatives]
            
            # Lookup main weights using pre-computed table indices
            main_weights = weights[table_indices_expanded, lookup_indices]  # [B, n_lookup_tables, n_outputs]
            
            # Lookup alt weights using pre-computed alternative indices
            alt_weights = weights[table_indices_expanded_alt, lookup_alt_indices]  # [B, n_lookup_tables, n_alternatives, n_outputs]
            
            # Weighted combination
            output = main_weights * main_weight.unsqueeze(2)  # [B, n_lookup_tables, n_outputs]
            alt_weighted = alt_weights * alt_weight.unsqueeze(3)  # [B, n_lookup_tables, n_alternatives, n_outputs]
            output = output + alt_weighted.sum(dim=2)  # [B, n_lookup_tables, n_outputs]
            
            # Save for backward (include gradient carriers)
            ctx.save_for_backward(
                weights, lookup_indices, lookup_alt_indices, lookup_alt_deltas,
                main_weight, alt_weight, lookup_indices_grad_c, lookup_alt_indices_grad_c,
                table_indices_flat, table_indices_alt_flat
            )
            ctx.smooth_mode = True
            ctx.batch_size = batch_size
            ctx.n_lookup_tables = n_lookup_tables
            ctx.n_alternatives = n_alternatives

        return (output,)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass.
        
        Propagates gradients to weights and lookup indices (via gradient carriers).
        """
        grad_output, = grad_outputs  # [B, n_lookup_tables, n_outputs]

        # Common input gradient computation (same for both modes).
        # Note: Uncertainty function is handled in AnchorPairsLookup backward,
        # so here we just do simple weight projections.
        def compute_input_gradients(grad_output_local, batch_size, n_lookup_tables, n_alternatives):
            # Gradient w.r.t. main lookup_indices: sum over output dimension
            lookup_indices_grad_local = grad_output_local.sum(dim=2)  # [B, n_lookup_tables]
            # Gradient w.r.t. alt lookup_indices: sum over output dimension
            grad_output_expanded_local = grad_output_local.unsqueeze(2).expand(
                batch_size, n_lookup_tables, n_alternatives, -1
            )  # [B, n_tables, n_alternatives, n_outputs]
            lookup_alt_indices_grad_local = grad_output_expanded_local.sum(dim=3)  # [B, n_tables, n_alternatives]
            return lookup_indices_grad_local, lookup_alt_indices_grad_local

        if not ctx.smooth_mode:
            # Non-smooth mode backward
            # Alternatives affect only input gradients, not weight gradients.
            (
                weights, lookup_indices, lookup_alt_indices,
                lookup_indices_grad_c, lookup_alt_indices_grad_c,
                table_indices_flat
            ) = ctx.saved_tensors
            batch_size = ctx.batch_size
            n_lookup_tables = ctx.n_lookup_tables
            n_alternatives = lookup_alt_indices_grad_c.shape[2]  # Gradient carriers are always provided

            # Gradient for weights (main indices only)
            weights_grad = torch.zeros_like(weights)
            weights_grad_flat = weights_grad.view(-1, weights_grad.shape[-1])  # [n_tables * n_entries, n_outputs]
            grad_output_flat = grad_output.reshape(batch_size * n_lookup_tables, grad_output.shape[-1])
            # table_indices_flat is passed via saved_tensors: [B * n_tables]
            lookup_indices_flat = lookup_indices.view(-1)                     # [B * n_tables]
            indices_flat = table_indices_flat * weights_grad.shape[1] + lookup_indices_flat

            weights_grad_flat.scatter_add_(0, indices_flat.unsqueeze(1).expand(-1, weights_grad.shape[-1]), grad_output_flat)
            weights_grad = weights_grad_flat.view(weights_grad.shape)

            # Gradients for lookup indices via gradient carriers (shared logic)
            lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad = compute_input_gradients(
                grad_output, batch_size, n_lookup_tables, n_alternatives
            )
            
            return weights_grad, None, None, None, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad, None, None, None, None, None, None
        else:
            # Smooth mode backward
            (
                weights, lookup_indices, lookup_alt_indices, lookup_alt_deltas,
                main_weight, alt_weight, lookup_indices_grad_c, lookup_alt_indices_grad_c,
                table_indices_flat, table_indices_alt_flat
            ) = ctx.saved_tensors
            
            batch_size = ctx.batch_size
            n_lookup_tables = ctx.n_lookup_tables
            n_alternatives = ctx.n_alternatives

            # Gradient for weights
            weights_grad = torch.zeros_like(weights)
            
            # Main weights gradient
            main_weight_expanded = main_weight.unsqueeze(2)  # [B, n_lookup_tables, 1]
            grad_main = grad_output * main_weight_expanded  # [B, n_lookup_tables, n_outputs]
            
            # Alt weights gradient
            alt_weight_expanded = alt_weight.unsqueeze(3)  # [B, n_lookup_tables, n_alternatives, 1]
            grad_output_expanded = grad_output.unsqueeze(2).expand(batch_size, n_lookup_tables, n_alternatives, -1)  # [B, n_lookup_tables, n_alternatives, n_outputs]
            grad_alt = grad_output_expanded * alt_weight_expanded  # [B, n_lookup_tables, n_alternatives, n_outputs]
            
            # Flatten for efficient scatter_add_
            weights_grad_flat = weights_grad.view(-1, weights_grad.shape[-1])  # [n_lookup_tables * n_entries_per_table, n_outputs]
            
            # Scatter main gradients using pre-computed table indices
            # table_indices_flat is passed via saved_tensors: [B * n_lookup_tables]
            lookup_indices_flat = lookup_indices.view(-1)  # [B * n_lookup_tables]
            indices_main = table_indices_flat * weights_grad.shape[1] + lookup_indices_flat  # [B * n_lookup_tables]
            grad_main_flat = grad_main.view(-1, grad_main.shape[-1])  # [B * n_lookup_tables, n_outputs]
            weights_grad_flat.scatter_add_(0, indices_main.unsqueeze(1).expand(-1, weights_grad.shape[-1]), grad_main_flat)
            
            # Scatter alt gradients using pre-computed alternative indices flat
            lookup_alt_indices_flat = lookup_alt_indices.view(-1)  # [B * n_lookup_tables * n_alternatives]
            indices_alt = table_indices_alt_flat * weights_grad.shape[1] + lookup_alt_indices_flat  # [B * n_lookup_tables * n_alternatives]
            grad_alt_flat = grad_alt.view(-1, grad_alt.shape[-1])  # [B * n_lookup_tables * n_alternatives, n_outputs]
            weights_grad_flat.scatter_add_(0, indices_alt.unsqueeze(1).expand(-1, weights_grad.shape[-1]), grad_alt_flat)
            
            weights_grad = weights_grad_flat.view(weights_grad.shape)
            
            # Gradients for lookup indices via gradient carriers (shared logic)
            lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad = compute_input_gradients(
                grad_output, batch_size, n_lookup_tables, n_alternatives
            )
            
            return weights_grad, None, None, None, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad, None, None, None, None, None, None
