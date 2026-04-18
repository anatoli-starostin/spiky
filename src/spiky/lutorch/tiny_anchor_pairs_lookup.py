"""
TinyAnchorPairsLookup: minimal anchor-pairs lookup for BitPermutationalLUT.

This is a stripped-down sibling of `AnchorPairsLookup`, built for bit-level
inference with fixed design choices.

Hardcoded constraints:
  - n_alternatives = 1 (single flip carrier for backward).
  - n_anchor_pairs <= 16 (primary lookup index fits in int16).
  - cmp_eps = 0 (exact comparison).
  - uncertainty_mode = INVERSE_L1, uncertainty_bias = 0.5.
  - anchor_sampling_policy = CANONICAL_DISTINCT (only).
  - shuffle_per_head = True.
  - smooth_mode = False.
  - No prebuilt_anchor_pairs, no anchor_candidates, no exclusion_sets,
    no connected_anchors_mode.

Forward signature:
  x: float [B, input_dim]
  returns (5-tuple):
    lookup_indices:         int16 [B, n_tables]
    lookup_alt_indices:     int16 [B, n_tables, 1]
    lookup_alt_deltas:      float [B, n_tables, 1]
    lookup_indices_grad_c:      float [B, n_tables]      (carrier for backward)
    lookup_alt_indices_grad_c:  float [B, n_tables, 1]   (carrier for backward)
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from spiky.lutorch.lut_helpers import (
    AnchorSamplingPolicy,
    get_balanced_anchor_pairs,
)

import os as _os

_USE_TINY_APL_CUSTOM_CUDA = _os.environ.get("SPIKY_TINY_APL_NO_CUSTOM_CUDA", "0") != "1"


def _get_tiny_apl_native():
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        return get_lutorch_manager()
    except Exception:
        return None


def _tiny_forward_pytorch(
    x: torch.Tensor,
    anchor_pairs_a: torch.Tensor,  # int16 [n_tables, n_anchor_pairs]
    anchor_pairs_b: torch.Tensor,
    powers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch implementation. Returns tuple
    (lookup_indices, lookup_alt_indices, lookup_alt_deltas, anchor1_ids, anchor2_ids).

    anchor_pairs_a/b are stored as int16; PyTorch gather requires int64 so we
    cast at the call site (cheap — constant tensors)."""
    batch_size = x.shape[0]
    n_tables, n_anchor_pairs = anchor_pairs_a.shape

    idx_a_long = anchor_pairs_a.long()  # cast once; gather needs int64
    idx_b_long = anchor_pairs_b.long()
    idx_a = idx_a_long.reshape(1, -1).expand(batch_size, -1)  # [B, n_tables*n_anchor_pairs]
    idx_b = idx_b_long.reshape(1, -1).expand(batch_size, -1)
    x_a = x.gather(1, idx_a).view(batch_size, n_tables, n_anchor_pairs)
    x_b = x.gather(1, idx_b).view(batch_size, n_tables, n_anchor_pairs)
    deltas = x_a - x_b  # float [B, n_tables, n_anchor_pairs]

    # Primary lookup index: bit_k = (delta_k > 0); sum as int32, cast to int16
    bits = deltas.gt(0.0).to(torch.int32)
    lookup_i32 = (bits << powers).sum(dim=2, dtype=torch.int32)  # [B, n_tables]

    # n_alternatives = 1: flip the single bit with smallest |delta|
    abs_deltas = deltas.abs()
    _, min_delta_indices = abs_deltas.min(dim=2, keepdim=True)  # long [B, n_tables, 1]
    lookup_alt_deltas = deltas.gather(2, min_delta_indices)      # float [B, n_tables, 1]

    flip_mask = (torch.ones(1, dtype=torch.int32, device=x.device) << min_delta_indices.to(torch.int32))
    lookup_alt_i32 = lookup_i32.unsqueeze(2) ^ flip_mask         # [B, n_tables, 1] int32

    lookup_indices = lookup_i32.to(torch.int16)
    lookup_alt_indices = lookup_alt_i32.to(torch.int16)

    # anchor1/2_ids as int16 (input_dim <= 32K) — saves memory during backward
    anchor1_ids = idx_a_long.unsqueeze(0).expand(batch_size, -1, -1).gather(2, min_delta_indices).to(torch.int16)
    anchor2_ids = idx_b_long.unsqueeze(0).expand(batch_size, -1, -1).gather(2, min_delta_indices).to(torch.int16)

    return lookup_indices, lookup_alt_indices, lookup_alt_deltas, anchor1_ids, anchor2_ids


def _tiny_backward_pytorch(
    x: torch.Tensor,
    anchor1_ids: torch.Tensor,  # int16 [B, n_tables, 1]
    anchor2_ids: torch.Tensor,
    lookup_alt_deltas: torch.Tensor,
    batch_offset: torch.Tensor,  # int32 [B * n_tables]
    grad_main: torch.Tensor,
    grad_alt: torch.Tensor,
) -> torch.Tensor:
    """INVERSE_L1 backward with uncertainty_bias=0.5 (fixed).

    anchor1/2_ids stored as int16; batch_offset as int32. Both are cast to
    int64 here only at the scatter_add_ call (PyTorch API requirement)."""
    batch_size, input_dim = x.shape

    grad_diff = grad_main.unsqueeze(2) - grad_alt  # [B, n_tables, 1]
    abs_delta = lookup_alt_deltas.abs()
    one_plus_abs = 1.0 + abs_delta
    minus_uncertainty_derivative = 0.5 * lookup_alt_deltas.sign() / (one_plus_abs * one_plus_abs)
    du = grad_diff * minus_uncertainty_derivative  # [B, n_tables, 1]

    x_grad_flat = torch.zeros(batch_size * input_dim, device=x.device, dtype=x.dtype)
    flat_a = (batch_offset + anchor1_ids.reshape(-1).to(torch.int32)).to(torch.int64)
    flat_b = (batch_offset + anchor2_ids.reshape(-1).to(torch.int32)).to(torch.int64)
    x_grad_flat.scatter_add_(0, flat_a, du.reshape(-1))
    x_grad_flat.scatter_add_(0, flat_b, -du.reshape(-1))
    return x_grad_flat


def _can_use_native_tiny_apl(x: torch.Tensor) -> bool:
    return (
        _USE_TINY_APL_CUSTOM_CUDA
        and x.is_cuda
        and x.dtype in (torch.float32, torch.float64)
        and _get_tiny_apl_native() is not None
    )


def _tiny_forward_cuda(x, anchor_pairs_a, anchor_pairs_b):
    """Returns (lookup_indices, lookup_alt_indices [B,n_tables,1],
    lookup_alt_deltas [B,n_tables,1], anchor1_ids [B,n_tables,1], anchor2_ids)."""
    native = _get_tiny_apl_native()
    lookup_idx, alt_idx, alt_delta, a1, a2 = native.tiny_apl_forward(
        x.contiguous(), anchor_pairs_a, anchor_pairs_b,
    )
    # Kernel returns collapsed [B, n_tables]; add the trailing n_alt=1 dim for API parity.
    return (
        lookup_idx,
        alt_idx.unsqueeze(-1),
        alt_delta.unsqueeze(-1),
        a1.unsqueeze(-1),
        a2.unsqueeze(-1),
    )


def _tiny_backward_cuda(x, anchor1_ids, anchor2_ids, lookup_alt_deltas,
                         grad_main, grad_alt, grad_direct=None):
    """Returns x_grad_flat [B * input_dim]."""
    native = _get_tiny_apl_native()
    B, input_dim = x.shape
    # Collapse trailing n_alt=1 dim for CUDA kernel.
    a1 = anchor1_ids.view(B, -1).contiguous()
    a2 = anchor2_ids.view(B, -1).contiguous()
    d = lookup_alt_deltas.view(B, -1).contiguous()
    ga = grad_alt.view(B, -1).contiguous()
    gd = grad_direct.view(B, -1).contiguous() if grad_direct is not None else None
    return native.tiny_apl_backward(
        int(B), int(input_dim),
        a1, a2, d, grad_main.contiguous(), ga, gd,
    )


class _TinyAnchorPairsLookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, anchor_pairs_a, anchor_pairs_b, powers, batch_offset):
        use_native = _can_use_native_tiny_apl(x)
        if use_native:
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                anchor1_ids,
                anchor2_ids,
            ) = _tiny_forward_cuda(x, anchor_pairs_a, anchor_pairs_b)
        else:
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                anchor1_ids,
                anchor2_ids,
            ) = _tiny_forward_pytorch(x, anchor_pairs_a, anchor_pairs_b, powers)

        n_tables = anchor_pairs_a.shape[0]
        batch_size = x.shape[0]

        # Carriers: zeros with autograd link to x. Downstream writes gradients here
        # which are harvested in backward().
        z = x.sum() * 0
        lookup_indices_grad_c = z.expand(batch_size, n_tables)
        lookup_alt_indices_grad_c = z.expand(batch_size, n_tables, 1)

        ctx.save_for_backward(x, anchor1_ids, anchor2_ids, lookup_alt_deltas, batch_offset)
        ctx.use_native = use_native
        return (
            lookup_indices,
            lookup_alt_indices,
            lookup_alt_deltas,
            lookup_indices_grad_c,
            lookup_alt_indices_grad_c,
        )

    @staticmethod
    def backward(ctx, grad_li, grad_lai, grad_lad, grad_lig, grad_laig):
        x, anchor1_ids, anchor2_ids, lookup_alt_deltas, batch_offset = ctx.saved_tensors
        if ctx.use_native:
            x_grad_flat = _tiny_backward_cuda(
                x, anchor1_ids, anchor2_ids, lookup_alt_deltas,
                grad_lig, grad_laig, grad_direct=grad_lad,
            )
        else:
            x_grad_flat = _tiny_backward_pytorch(
                x, anchor1_ids, anchor2_ids, lookup_alt_deltas, batch_offset,
                grad_lig, grad_laig,
            )
            if grad_lad is not None:
                d = grad_lad.reshape(-1)
                flat_a = (batch_offset + anchor1_ids.reshape(-1).to(torch.int32)).to(torch.int64)
                flat_b = (batch_offset + anchor2_ids.reshape(-1).to(torch.int32)).to(torch.int64)
                x_grad_flat.scatter_add_(0, flat_a, d)
                x_grad_flat.scatter_add_(0, flat_b, -d)
        return x_grad_flat.view(x.shape), None, None, None, None


class TinyAnchorPairsLookup(nn.Module):
    """Minimal anchor-pairs lookup (see module docstring).

    Inputs:
      x: float [B, input_dim]
    Outputs: 5-tuple with int16 indices + float carriers (see module docstring).
    """

    def __init__(
        self,
        input_dim: int,
        n_tables: int,
        n_anchor_pairs: int,
        n_heads: int = 1,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not (1 <= n_anchor_pairs <= 16):
            raise ValueError(
                f"TinyAnchorPairsLookup requires 1 <= n_anchor_pairs <= 16, "
                f"got {n_anchor_pairs}"
            )
        self.input_dim = input_dim
        self.n_tables = n_tables
        self.n_anchor_pairs = n_anchor_pairs
        self.table_dim = 1 << n_anchor_pairs  # 2 ** n_anchor_pairs

        dev = device or torch.device("cpu")
        anchor_pairs_a, anchor_pairs_b = get_balanced_anchor_pairs(
            n_tables=n_tables,
            n_anchor_pairs=n_anchor_pairs,
            input_dim=input_dim,
            device=dev,
            random_seed=random_seed,
            policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
            n_heads=n_heads,
            shuffle_per_head=True,
        )
        if input_dim > 32767:
            raise ValueError(
                f"TinyAnchorPairsLookup requires input_dim <= 32767 (int16 limit), got {input_dim}"
            )
        self.register_buffer("anchor_pairs_a", anchor_pairs_a.contiguous().to(torch.int16))
        self.register_buffer("anchor_pairs_b", anchor_pairs_b.contiguous().to(torch.int16))

        # Bit positions used as left-shifts when assembling the lookup index.
        powers = torch.arange(n_anchor_pairs, dtype=torch.int32, device=dev).view(1, 1, -1)
        self.register_buffer("powers", powers)

        # Cache for batch_offset used in backward scatter (int32 — cast to int64 at use-site).
        self._cached_batch_offset: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor):
        if not x.is_floating_point():
            raise TypeError(f"TinyAnchorPairsLookup input must be float, got {x.dtype}")
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        batch_size = x.shape[0]
        expected_len = batch_size * self.n_tables  # n_alternatives = 1
        if (
            self._cached_batch_offset is None
            or self._cached_batch_offset.numel() != expected_len
            or self._cached_batch_offset.device != x.device
        ):
            self._cached_batch_offset = (
                torch.arange(batch_size, device=x.device, dtype=torch.int32)
                .repeat_interleave(self.n_tables) * self.input_dim
            ).contiguous()

        if x.requires_grad or self.training:
            return _TinyAnchorPairsLookupFunction.apply(
                x, self.anchor_pairs_a, self.anchor_pairs_b, self.powers,
                self._cached_batch_offset,
            )
        # Eval no-grad path: skip autograd Function overhead.
        if _can_use_native_tiny_apl(x):
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                _,
                _,
            ) = _tiny_forward_cuda(x, self.anchor_pairs_a, self.anchor_pairs_b)
        else:
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                _,
                _,
            ) = _tiny_forward_pytorch(x, self.anchor_pairs_a, self.anchor_pairs_b, self.powers)
        return lookup_indices, lookup_alt_indices, lookup_alt_deltas, None, None
