"""
LProjection module for lookup table-based projection.
"""
import os
import torch
import torch.nn as nn
from typing import Tuple, Optional

from spiky.lutorch.lut_helpers import UncertaintyMode


# Optional torch.compile; set SPIKY_LUTORCH_NO_COMPILE=1 to disable (e.g. debugging or older PyTorch).
_USE_LUTORCH_COMPILE = os.environ.get("SPIKY_LUTORCH_NO_COMPILE", "0") != "1"

# Custom CUDA kernels can be disabled globally for all LUTorch ops via
# SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS=1, and independently for LProjection via
# SPIKY_LUTORCH_NO_LPROJECTION_CUSTOM_CUDA_KERNELS=1.
_USE_LUTORCH_CUSTOM_CUDA_KERNELS_GLOBAL = os.environ.get("SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS", "0") != "1"
_USE_LUTORCH_LPROJECTION_CUSTOM_CUDA_KERNELS = os.environ.get(
    "SPIKY_LUTORCH_NO_LPROJECTION_CUSTOM_CUDA_KERNELS", "0"
) != "1"
_USE_LUTORCH_CUSTOM_CUDA_KERNELS = (
    _USE_LUTORCH_CUSTOM_CUDA_KERNELS_GLOBAL and _USE_LUTORCH_LPROJECTION_CUSTOM_CUDA_KERNELS
)
_LUTORCH_CUDA_THREADS_PER_BLOCK = int(os.environ.get("SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK", "256"))
if _LUTORCH_CUDA_THREADS_PER_BLOCK < 1 or _LUTORCH_CUDA_THREADS_PER_BLOCK > 1024:
    raise ValueError(
        "SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK must be in range [1, 1024], "
        f"got {_LUTORCH_CUDA_THREADS_PER_BLOCK}"
    )

try:
    from spiky_cuda import get_lutorch_manager as _get_native_lutorch_manager
except Exception:
    def _get_native_lutorch_manager():
        return None


def _maybe_compile(fn):
    if _USE_LUTORCH_COMPILE and hasattr(torch, "compile"):
        return torch.compile(fn, dynamic=True)
    return fn


# --- Compiled hot-path implementations (torch.compile for fusion when enabled) ---


@_maybe_compile
def _forward_smooth_impl(
    weights: torch.Tensor,
    table_indices_expanded: torch.Tensor,
    table_indices_expanded_alt: torch.Tensor,
    lookup_indices: torch.Tensor,
    lookup_alt_indices: torch.Tensor,
    lookup_alt_deltas: torch.Tensor,
    n_alternatives: int,
    l1_uncertainty: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth forward branch, returns (output, main_weight, alt_weight)."""
    if l1_uncertainty:
        abs_deltas = lookup_alt_deltas.abs()
        uncertainty = 0.5 / (1.0 + abs_deltas)
    else:
        squared = lookup_alt_deltas * lookup_alt_deltas
        uncertainty = 0.5 / (1.0 + squared)

    main_weight = 1.0 - (uncertainty.sum(dim=2) / n_alternatives)
    alt_weight = uncertainty / n_alternatives

    main_weights = weights[table_indices_expanded, lookup_indices]
    alt_weights = weights[table_indices_expanded_alt, lookup_alt_indices]

    output = main_weights * main_weight.unsqueeze(2)
    output = output + (alt_weights * alt_weight.unsqueeze(3)).sum(dim=2)
    return output, main_weight, alt_weight


@_maybe_compile
def _backward_train_impl(
    grad_output: torch.Tensor,
    weights: torch.Tensor,
    lookup_indices: torch.Tensor,
    table_indices_flat: torch.Tensor,
    batch_size: int,
    n_lookup_tables: int,
    n_alternatives: int,
    main_weight: Optional[torch.Tensor],
    alt_weight: Optional[torch.Tensor],
    lookup_alt_indices: torch.Tensor,
    table_indices_alt_flat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train backward (smooth-style): weights gradient via scatter_add_, and input gradient carriers.
    Non-smooth mode: main_weight=None, alt_weight=None → treat as main=1.0, alt=0.0.
    """
    n_entries = weights.shape[1]
    n_outputs = weights.shape[2]
    weights_grad = torch.zeros_like(weights)
    weights_grad_flat = weights_grad.view(-1, n_outputs)

    grad_output_expanded = grad_output.unsqueeze(2).expand(
        batch_size, n_lookup_tables, n_alternatives, -1
    )

    if main_weight is None:
        grad_main = grad_output
    else:
        main_weight_expanded = main_weight.unsqueeze(2)
        grad_main = grad_output * main_weight_expanded
        alt_weight_expanded = alt_weight.unsqueeze(3)
        grad_alt = grad_output_expanded * alt_weight_expanded

    lookup_indices_flat = lookup_indices.view(-1)
    indices_main = table_indices_flat * n_entries + lookup_indices_flat
    grad_main_flat = grad_main.reshape(-1, n_outputs)
    weights_grad_flat.scatter_add_(
        0, indices_main.unsqueeze(1).expand(-1, n_outputs), grad_main_flat
    )

    if main_weight is not None:
        lookup_alt_indices_flat = lookup_alt_indices.view(-1)
        indices_alt = table_indices_alt_flat * n_entries + lookup_alt_indices_flat
        grad_alt_flat = grad_alt.reshape(-1, n_outputs)
        weights_grad_flat.scatter_add_(
            0, indices_alt.unsqueeze(1).expand(-1, n_outputs), grad_alt_flat
        )

    weights_grad = weights_grad_flat.view(weights.shape)

    table_indices_2d = table_indices_flat.view(batch_size, n_lookup_tables)
    main_weights = weights[table_indices_2d, lookup_indices]
    lookup_indices_grad_c_grad = (grad_output * main_weights).sum(dim=2)

    table_indices_alt_3d = table_indices_alt_flat.view(batch_size, n_lookup_tables, n_alternatives)
    lookup_alt_indices_3d = lookup_alt_indices.view(batch_size, n_lookup_tables, n_alternatives)
    alt_weights = weights[table_indices_alt_3d, lookup_alt_indices_3d]
    lookup_alt_indices_grad_c_grad = (grad_output_expanded * alt_weights).sum(dim=3)

    return weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad


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
        device: Optional[torch.device] = None,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
    ):
        super().__init__()
        self.n_lookup_tables = n_lookup_tables
        self.n_entries_per_table = n_entries_per_table
        self.n_alternatives = n_alternatives
        self.n_outputs = n_outputs
        self.smooth_mode = smooth_mode
        self.uncertainty_mode = uncertainty_mode
        
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

            l1_uncertainty = self.uncertainty_mode == UncertaintyMode.INVERSE_L1

            use_native_eval_cuda = (
                _USE_LUTORCH_CUSTOM_CUDA_KERNELS
                and _get_native_lutorch_manager() is not None
                and self.weights.is_cuda
                and self.weights.dtype in (torch.float32, torch.float64)
            )
            if use_native_eval_cuda:
                native = _get_native_lutorch_manager()
                output, _, _ = native.lprojection_forward_smooth(
                    self.weights,
                    lookup_indices.contiguous(),
                    lookup_alt_indices.contiguous(),
                    lookup_alt_deltas.contiguous(),
                    self._cached_table_indices_flat.contiguous(),
                    self._cached_table_indices_alt_flat.contiguous(),
                    l1_uncertainty,
                    _LUTORCH_CUDA_THREADS_PER_BLOCK,
                )
                return output

            output, _, _ = _forward_smooth_impl(
                self.weights,
                self._cached_table_indices_expanded,
                self._cached_table_indices_expanded_alt,
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                self.n_alternatives,
                l1_uncertainty
            )
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
            self.smooth_mode, self.n_alternatives, self.uncertainty_mode,
            self._cached_table_indices_expanded, self._cached_table_indices_flat,
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
                   lookup_indices_grad_c, lookup_alt_indices_grad_c, smooth_mode,
                   n_alternatives, uncertainty_mode, table_indices_expanded, table_indices_flat,
                   table_indices_expanded_alt, table_indices_alt_flat
        
        Returns:
            Output tensor of shape [B, n_lookup_tables, n_outputs]
        """
        (
            weights, lookup_indices, lookup_alt_indices, lookup_alt_deltas,
            _, _, smooth_mode,
            n_alternatives, uncertainty_mode, table_indices_expanded, table_indices_flat,
            table_indices_expanded_alt, table_indices_alt_flat
        ) = args
        
        batch_size = lookup_indices.shape[0]
        n_lookup_tables = lookup_indices.shape[1]

        if not smooth_mode:
            # Non-smooth: just lookup weights using pre-computed table indices
            output = weights[table_indices_expanded, lookup_indices]  # [B, n_lookup_tables, n_outputs]
            # Non-smooth: main_weight=None, alt_weight=None (backward treats as 1.0 and 0.0)
            ctx.save_for_backward(
                weights, lookup_indices, lookup_alt_indices,
                None, None, table_indices_flat, table_indices_alt_flat
            )
            ctx.smooth_mode = False
            ctx.batch_size = batch_size
            ctx.n_lookup_tables = n_lookup_tables
            ctx.n_alternatives = n_alternatives
        else:
            # Smooth mode: weighted combination
            assert lookup_alt_indices is not None, "lookup_alt_indices required in smooth mode"
            assert lookup_alt_deltas is not None, "lookup_alt_deltas required in smooth mode"
            l1_uncertainty = uncertainty_mode == UncertaintyMode.INVERSE_L1
            use_native_forward_cuda = (
                _USE_LUTORCH_CUSTOM_CUDA_KERNELS
                and _get_native_lutorch_manager() is not None
                and weights.is_cuda
                and weights.dtype in (torch.float32, torch.float64)
            )
            if use_native_forward_cuda:
                native = _get_native_lutorch_manager()
                output, main_weight, alt_weight = native.lprojection_forward_smooth(
                    weights,
                    lookup_indices.contiguous(),
                    lookup_alt_indices.contiguous(),
                    lookup_alt_deltas.contiguous(),
                    table_indices_flat.contiguous(),
                    table_indices_alt_flat.contiguous(),
                    l1_uncertainty,
                    _LUTORCH_CUDA_THREADS_PER_BLOCK,
                )
            else:
                output, main_weight, alt_weight = _forward_smooth_impl(
                    weights,
                    table_indices_expanded,
                    table_indices_expanded_alt,
                    lookup_indices,
                    lookup_alt_indices,
                    lookup_alt_deltas,
                    n_alternatives,
                    l1_uncertainty,
                )
            ctx.save_for_backward(
                weights, lookup_indices, lookup_alt_indices,
                main_weight, alt_weight, table_indices_flat,
                table_indices_alt_flat
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
        Same path for smooth and non-smooth; non-smooth has main_weight=None, alt_weight=None.
        """
        grad_output, = grad_outputs  # [B, n_lookup_tables, n_outputs]

        (
            weights, lookup_indices, lookup_alt_indices,
            main_weight, alt_weight, table_indices_flat, table_indices_alt_flat
        ) = ctx.saved_tensors

        batch_size = ctx.batch_size
        n_lookup_tables = ctx.n_lookup_tables
        n_alternatives = ctx.n_alternatives

        use_native_cuda_backward = (
            _USE_LUTORCH_CUSTOM_CUDA_KERNELS
            and _get_native_lutorch_manager() is not None
            and grad_output.is_cuda
            and grad_output.dtype in (torch.float32, torch.float64)
            and lookup_alt_indices is not None
        )
        if use_native_cuda_backward:
            native = _get_native_lutorch_manager()
            lookup_indices_c = lookup_indices.contiguous()
            lookup_alt_indices_c = lookup_alt_indices.contiguous()
            table_indices_flat_c = table_indices_flat.contiguous()
            table_indices_alt_flat_c = table_indices_alt_flat.contiguous()
            if ctx.smooth_mode:
                main_weight_c = main_weight.contiguous()
                alt_weight_c = alt_weight.contiguous()
                if n_alternatives == 1:
                    weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad = native.lprojection_backward_na1_smooth(
                        grad_output,
                        weights,
                        lookup_indices_c,
                        lookup_alt_indices_c,
                        table_indices_flat_c,
                        table_indices_alt_flat_c,
                        main_weight_c,
                        alt_weight_c,
                        _LUTORCH_CUDA_THREADS_PER_BLOCK,
                    )
                else:
                    weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad = native.lprojection_backward_smooth(
                        grad_output,
                        weights,
                        lookup_indices_c,
                        lookup_alt_indices_c,
                        table_indices_flat_c,
                        table_indices_alt_flat_c,
                        main_weight_c,
                        alt_weight_c,
                        _LUTORCH_CUDA_THREADS_PER_BLOCK,
                    )
            else:
                if n_alternatives == 1:
                    weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad = native.lprojection_backward_na1_nonsmooth(
                        grad_output,
                        weights,
                        lookup_indices_c,
                        lookup_alt_indices_c,
                        table_indices_flat_c,
                        table_indices_alt_flat_c,
                        _LUTORCH_CUDA_THREADS_PER_BLOCK,
                    )
                else:
                    weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad = native.lprojection_backward_nonsmooth(
                        grad_output,
                        weights,
                        lookup_indices_c,
                        lookup_alt_indices_c,
                        table_indices_flat_c,
                        table_indices_alt_flat_c,
                        _LUTORCH_CUDA_THREADS_PER_BLOCK,
                    )
        else:
            weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad = _backward_train_impl(
                grad_output, weights, lookup_indices, table_indices_flat,
                batch_size, n_lookup_tables, n_alternatives,
                main_weight, alt_weight, lookup_alt_indices, table_indices_alt_flat,
            )

        return (
            weights_grad,
            None,
            None,
            None,
            lookup_indices_grad_c_grad,
            lookup_alt_indices_grad_c_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
