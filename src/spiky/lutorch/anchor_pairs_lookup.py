"""
Anchor pairs lookup implementation.
"""
import os
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from spiky.lutorch.abstract_lookup import AbstractLookup
from spiky.lutorch.anchor_sampler import AnchorSampler
from spiky.lutorch.lut_helpers import UncertaintyMode, get_balanced_anchor_pairs
from spiky.util.chunk_of_connections import ChunkOfConnections

# Optional torch.compile; set SPIKY_LUTORCH_NO_COMPILE=1 to disable (e.g. debugging or older PyTorch).
_USE_LUTORCH_COMPILE = os.environ.get("SPIKY_LUTORCH_NO_COMPILE", "0") != "1"
# Optional native CUDA kernels in lutorch; set SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS=1 to disable.
_USE_LUTORCH_CUSTOM_CUDA_KERNELS = os.environ.get("SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS", "0") != "1"
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


@_maybe_compile
def _anchor_pairs_lookup_forward_fallback(
    x: torch.Tensor,
    anchor_pairs_a: torch.Tensor,
    anchor_pairs_b: torch.Tensor,
    powers: torch.Tensor,
    cmp_eps: float,
    n_alternatives: int,
):
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

    if n_alternatives == n_anchor_pairs:
        # All positions: no topk, min_delta_indices = 0..n_anchor_pairs-1, lookup_alt_deltas = deltas
        min_delta_indices = torch.arange(
            n_anchor_pairs, device=x.device, dtype=torch.long
        ).view(1, 1, -1).expand(batch_size, n_tables, -1)
        lookup_alt_deltas = deltas
        anchor1_ids = anchor_pairs_a.unsqueeze(0).repeat(batch_size, 1, 1)
        anchor2_ids = anchor_pairs_b.unsqueeze(0).repeat(batch_size, 1, 1)
    else:
        abs_deltas = deltas.abs()  # [B, n_tables, n_anchor_pairs]
        if n_alternatives == 1:
            _, min_delta_indices = abs_deltas.min(dim=2, keepdim=True)  # [B, n_tables, 1]
            lookup_alt_deltas = deltas.gather(2, min_delta_indices)
        else:
            min_delta_indices = torch.topk(
                abs_deltas, k=n_alternatives, dim=2, largest=False
            ).indices  # [B, n_tables, n_alternatives]
            lookup_alt_deltas = deltas.gather(2, min_delta_indices)
        anchor1_ids = anchor_pairs_a.unsqueeze(0).expand(batch_size, -1, -1).gather(2, min_delta_indices)
        anchor2_ids = anchor_pairs_b.unsqueeze(0).expand(batch_size, -1, -1).gather(2, min_delta_indices)

    lookup_indices_expanded = lookup_indices.unsqueeze(2)
    flip_masks = (1 << min_delta_indices).long()
    lookup_alt_indices = (lookup_indices_expanded ^ flip_masks)

    return lookup_indices, lookup_alt_indices, lookup_alt_deltas, anchor1_ids, anchor2_ids


@_maybe_compile
def _anchor_pairs_lookup_eval_fallback(
    x: torch.Tensor,
    anchor_pairs_a: torch.Tensor,
    anchor_pairs_b: torch.Tensor,
    powers: torch.Tensor,
    cmp_eps: float,
    n_tables: int,
    n_anchor_pairs: int,
    n_alternatives: int,
    return_alternatives: bool,
):
    batch_size = x.shape[0]
    idx_a = anchor_pairs_a.reshape(1, -1).expand(batch_size, -1)  # [B, n_tables*n_anchor_pairs]
    idx_b = anchor_pairs_b.reshape(1, -1).expand(batch_size, -1)
    x_a = x.gather(1, idx_a).view(batch_size, n_tables, n_anchor_pairs)
    x_b = x.gather(1, idx_b).view(batch_size, n_tables, n_anchor_pairs)
    deltas = x_a - x_b

    bits = deltas.gt(cmp_eps).to(dtype=torch.long)
    lookup_indices = (bits << powers).sum(dim=2, dtype=torch.long)  # [B, n_tables]

    if return_alternatives:
        if n_alternatives == n_anchor_pairs:
            min_delta_indices = torch.arange(
                n_anchor_pairs, device=x.device, dtype=torch.long
            ).view(1, 1, -1).expand(batch_size, n_tables, -1)
            lookup_alt_deltas = deltas
        else:
            abs_deltas = deltas.abs()  # [B, n_tables, n_anchor_pairs]
            if n_alternatives == 1:
                _, min_delta_indices = abs_deltas.min(dim=2, keepdim=True)
                lookup_alt_deltas = deltas.gather(2, min_delta_indices)
            else:
                min_delta_indices = torch.topk(
                    abs_deltas, k=n_alternatives, dim=2, largest=False
                ).indices
                lookup_alt_deltas = deltas.gather(2, min_delta_indices)
        lookup_indices_expanded = lookup_indices.unsqueeze(2)
        flip_masks = (1 << min_delta_indices).long()
        lookup_alt_indices = lookup_indices_expanded ^ flip_masks
    else:
        lookup_alt_indices = None
        lookup_alt_deltas = None

    return lookup_indices, lookup_alt_indices, lookup_alt_deltas


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
        connected_anchors_mode: If True, anchor pairs form a connected graph (flat_b = flat_a shifted by 1 in balanced path).
        anchor_candidates: Optional. Either:
                          - torch.Tensor: Shape [n_tables, max_anchors_per_table] with input indices
                            (all values must be >= 0, no padding)
                          - Tuple[ChunkOfConnections, int]: ChunkOfConnections with custom ids_shift
                          - None: use balanced coverage (get_balanced_anchor_pairs); otherwise use AnchorSampler
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
        connected_anchors_mode: bool = False,
        anchor_candidates: Optional[Union[torch.Tensor, Tuple[ChunkOfConnections, int]]] = None,
        cmp_eps: float = 0.0,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        n_alternatives: int = 1,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
    ):
        table_dim = 2 ** n_anchor_pairs
        if n_alternatives > n_anchor_pairs:
            raise ValueError(
                f"n_alternatives ({n_alternatives}) must be <= n_anchor_pairs ({n_anchor_pairs})"
            )
        super().__init__(input_dim, n_tables, table_dim, n_alternatives=n_alternatives)

        self.n_anchor_pairs = n_anchor_pairs
        assert cmp_eps >= 0.0
        self.cmp_eps = cmp_eps
        self.uncertainty_mode = uncertainty_mode

        dev = device or torch.device("cpu")
        if anchor_candidates is None:
            # Balanced coverage over input dimensions (randperm-based)
            anchor_pairs_a, anchor_pairs_b = get_balanced_anchor_pairs(
                n_tables, n_anchor_pairs, input_dim, dev,
                random_seed=random_seed,
                connected_mode=connected_anchors_mode,
            )
            self.register_buffer("anchor_pairs_a", anchor_pairs_a.contiguous())
            self.register_buffer("anchor_pairs_b", anchor_pairs_b.contiguous())
        else:
            # Use AnchorSampler when explicit candidate connections are given
            anchor_sampler = AnchorSampler(
                n_inputs=input_dim,
                n_detectors=n_tables,
                n_anchors_per_detector=n_anchor_pairs,
                connected_anchors_mode=connected_anchors_mode,
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
                - lookup_alt_deltas: float [B, n_tables, n_alternatives]
                - lookup_indices_grad_c: float [B, n_tables]
                - lookup_alt_indices_grad_c: float [B, n_tables, n_alternatives]
            In eval mode:
                - lookup_indices: int64 [B, n_tables]
                - lookup_alt_indices: int64 [B, n_tables, n_alternatives] (or empty if return_alternatives=False)
        """
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
        use_native_eval_cuda = (
            _USE_LUTORCH_CUSTOM_CUDA_KERNELS
            and
            self.n_alternatives in (1, 2, 3)
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float64)
            and _get_native_lutorch_manager() is not None
        )
        if use_native_eval_cuda:
            native = _get_native_lutorch_manager()
            if return_alternatives:
                if self.n_alternatives == 1:
                    lookup_indices, lookup_alt_indices, lookup_alt_deltas, _, _ = native.anchor_pairs_lookup_forward_na1(
                        x,
                        anchor_pairs_a,
                        anchor_pairs_b,
                        float(self.cmp_eps),
                        False,
                        _LUTORCH_CUDA_THREADS_PER_BLOCK,
                    )
                elif self.n_alternatives == 2:
                    lookup_indices, lookup_alt_indices, lookup_alt_deltas, _, _ = native.anchor_pairs_lookup_forward_na2(
                        x,
                        anchor_pairs_a,
                        anchor_pairs_b,
                        float(self.cmp_eps),
                        False,
                        _LUTORCH_CUDA_THREADS_PER_BLOCK,
                    )
                else:
                    lookup_indices, lookup_alt_indices, lookup_alt_deltas, _, _ = native.anchor_pairs_lookup_forward_na3(
                        x,
                        anchor_pairs_a,
                        anchor_pairs_b,
                        float(self.cmp_eps),
                        False,
                        _LUTORCH_CUDA_THREADS_PER_BLOCK,
                    )
                return lookup_indices, lookup_alt_indices, lookup_alt_deltas
            lookup_indices = native.anchor_pairs_lookup_eval_forward(
                x,
                anchor_pairs_a,
                anchor_pairs_b,
                float(self.cmp_eps),
                _LUTORCH_CUDA_THREADS_PER_BLOCK,
            )
            return lookup_indices, None, None

        return _anchor_pairs_lookup_eval_fallback(
            x,
            anchor_pairs_a,
            anchor_pairs_b,
            self.powers,
            self.cmp_eps,
            self.n_tables,
            self.n_anchor_pairs,
            self.n_alternatives,
            return_alternatives,
        )

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
        n_tables = anchor_pairs_a.shape[0]

        use_native_cuda = (
            _USE_LUTORCH_CUSTOM_CUDA_KERNELS
            and
            n_alternatives in (1, 2, 3)
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float64)
            and _get_native_lutorch_manager() is not None
        )
        if use_native_cuda:
            native = _get_native_lutorch_manager()
            if n_alternatives == 1:
                (
                    lookup_indices,
                    lookup_alt_indices,
                    lookup_alt_deltas,
                    anchor1_ids,
                    anchor2_ids,
                ) = native.anchor_pairs_lookup_forward_na1(
                    x,
                    anchor_pairs_a,
                    anchor_pairs_b,
                    float(cmp_eps),
                    True,
                    _LUTORCH_CUDA_THREADS_PER_BLOCK,
                )
            elif n_alternatives == 2:
                (
                    lookup_indices,
                    lookup_alt_indices,
                    lookup_alt_deltas,
                    anchor1_ids,
                    anchor2_ids,
                ) = native.anchor_pairs_lookup_forward_na2(
                    x,
                    anchor_pairs_a,
                    anchor_pairs_b,
                    float(cmp_eps),
                    True,
                    _LUTORCH_CUDA_THREADS_PER_BLOCK,
                )
            else:
                (
                    lookup_indices,
                    lookup_alt_indices,
                    lookup_alt_deltas,
                    anchor1_ids,
                    anchor2_ids,
                ) = native.anchor_pairs_lookup_forward_na3(
                    x,
                    anchor_pairs_a,
                    anchor_pairs_b,
                    float(cmp_eps),
                    True,
                    _LUTORCH_CUDA_THREADS_PER_BLOCK,
                )
        else:
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                anchor1_ids,
                anchor2_ids,
            ) = _anchor_pairs_lookup_forward_fallback(
                x,
                anchor_pairs_a,
                anchor_pairs_b,
                powers,
                cmp_eps,
                n_alternatives,
            )

        z = x.sum() * 0
        lookup_indices_grad_c = z.expand(batch_size, n_tables)
        lookup_alt_indices_grad_c = z.expand(batch_size, n_tables, n_alternatives)

        ctx.inv_l1 = (int(uncertainty_mode) == 0)
        ctx.batch_offset = batch_offset
        ctx.save_for_backward(x, anchor1_ids, anchor2_ids, lookup_alt_deltas)

        return (
            lookup_indices,
            lookup_alt_indices,
            lookup_alt_deltas,
            lookup_indices_grad_c,
            lookup_alt_indices_grad_c
        )

    @staticmethod
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

        use_native_backward_cuda = (
            _USE_LUTORCH_CUSTOM_CUDA_KERNELS
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float64)
            and _get_native_lutorch_manager() is not None
            and lookup_alt_deltas.shape[-1] in (1, 2, 3)
            and anchor1_ids.numel() > 0
            and anchor2_ids.numel() > 0
        )
        if use_native_backward_cuda:
            native = _get_native_lutorch_manager()
            if lookup_alt_deltas.shape[-1] == 1:
                x_grad_flat = native.anchor_pairs_lookup_backward_na1(
                    x,
                    anchor1_ids.reshape(-1).contiguous(),
                    anchor2_ids.reshape(-1).contiguous(),
                    lookup_alt_deltas.reshape(-1).contiguous(),
                    ctx.batch_offset.reshape(-1).contiguous(),
                    grad_lookup_indices_grad_c,
                    grad_lookup_alt_indices_grad_c.reshape(-1).contiguous(),
                    ctx.inv_l1,
                    _LUTORCH_CUDA_THREADS_PER_BLOCK,
                )
            elif lookup_alt_deltas.shape[-1] == 2:
                x_grad_flat = native.anchor_pairs_lookup_backward_na2(
                    x,
                    anchor1_ids.reshape(-1).contiguous(),
                    anchor2_ids.reshape(-1).contiguous(),
                    lookup_alt_deltas.reshape(-1).contiguous(),
                    ctx.batch_offset.reshape(-1).contiguous(),
                    grad_lookup_indices_grad_c,
                    grad_lookup_alt_indices_grad_c.reshape(-1).contiguous(),
                    ctx.inv_l1,
                    _LUTORCH_CUDA_THREADS_PER_BLOCK,
                )
            else:
                x_grad_flat = native.anchor_pairs_lookup_backward_na3(
                    x,
                    anchor1_ids.reshape(-1).contiguous(),
                    anchor2_ids.reshape(-1).contiguous(),
                    lookup_alt_deltas.reshape(-1).contiguous(),
                    ctx.batch_offset.reshape(-1).contiguous(),
                    grad_lookup_indices_grad_c,
                    grad_lookup_alt_indices_grad_c.reshape(-1).contiguous(),
                    ctx.inv_l1,
                    _LUTORCH_CUDA_THREADS_PER_BLOCK,
                )
            return x_grad_flat.view(x.shape), None, None, None, None, None, None, None

        def _anchor_pairs_lookup_backward_impl(
            x,
            anchor1_ids,
            anchor2_ids,
            lookup_alt_deltas,
            ctx_inv_l1,
            ctx_batch_offset,
            grad_main,
            grad_alt,
        ):
            batch_size, input_dim = x.shape[0], x.shape[1]

            grad_diff = grad_main.unsqueeze(2) - grad_alt

            if ctx_inv_l1:
                abs_delta = lookup_alt_deltas.abs()
                one_plus_abs = 1.0 + abs_delta
                minus_uncertainty_derivative = 0.5 * lookup_alt_deltas.sign() / (one_plus_abs * one_plus_abs)
            else:
                delta_sq = lookup_alt_deltas * lookup_alt_deltas
                one_plus_sq = 1.0 + delta_sq
                minus_uncertainty_derivative = lookup_alt_deltas / (one_plus_sq * one_plus_sq)

            du = grad_diff * minus_uncertainty_derivative  # [B, n_tables, n_alternatives]
            if lookup_alt_deltas.shape[-1] > 1:
                du /= lookup_alt_deltas.shape[-1]

            batch_offset = ctx_batch_offset
            anchor1_flat = anchor1_ids.view(-1)
            anchor2_flat = anchor2_ids.view(-1)
            du_flat = du.view(-1)
            x_grad_flat = torch.zeros(batch_size * input_dim, device=x.device, dtype=x.dtype)
            indices1 = batch_offset + anchor1_flat
            indices2 = batch_offset + anchor2_flat
            x_grad_flat.scatter_add_(0, indices1, du_flat)
            x_grad_flat.scatter_add_(0, indices2, -du_flat)
            return x_grad_flat

        _anchor_pairs_lookup_backward_impl = _maybe_compile(_anchor_pairs_lookup_backward_impl)

        x_grad_flat = _anchor_pairs_lookup_backward_impl(
            x,
            anchor1_ids,
            anchor2_ids,
            lookup_alt_deltas,
            ctx.inv_l1,
            ctx.batch_offset,
            grad_lookup_indices_grad_c,
            grad_lookup_alt_indices_grad_c,
        )

        # 8 inputs -> 8 gradient returns
        return x_grad_flat.view(x.shape), None, None, None, None, None, None, None
