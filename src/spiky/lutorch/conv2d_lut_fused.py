"""
Conv2DLutFused — fused version of Conv2DLut that avoids the F.unfold
intermediate tensor.

Instead of materialising a [B, C*kH*kW, n_patches] unfolded tensor,
new CUDA kernels read directly from the 4D input [B, C, H, W] by
decoding anchor flat indices on-the-fly:
    anchor_flat ∈ [0, C*kH*kW)
    c  = anchor_flat // (kH*kW)
    kr = (anchor_flat % (kH*kW)) // kW
    kc = anchor_flat % kW
    h_in = h_out*sH + kr - pH
    w_in = w_out*sW + kc - pW

Only n_alternatives=3 and smooth_mode=True are implemented (the only
combination used in practice).  For other configs the module falls back
to the reference Conv2DLut transparently.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from spiky.lutorch.multi_head_lut import Conv2DLut, UnfoldConfiguration
from spiky.lutorch.l_projection import LProjection
from spiky.lutorch.lut_helpers import UncertaintyMode, get_balanced_anchor_pairs


# Honour the same env-var knob as the rest of LUTorch.
_USE_LUTORCH_CUSTOM_CUDA_KERNELS = (
    os.environ.get("SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS", "0") != "1"
)
_LUTORCH_CUDA_THREADS_PER_BLOCK = int(os.environ.get("SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK", "256"))


def _get_native_lutorch_manager():
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        return get_lutorch_manager()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Autograd function for the fused conv2d anchor-pairs lookup (na3)
# ---------------------------------------------------------------------------

class Conv2DAnchorPairsLookupFunctionNa3(torch.autograd.Function):
    """
    Fused forward+backward for conv2d anchor pairs lookup with na3.

    Forward outputs five tensors that are consumed by LProjection:
        lookup_indices         [B*H_out*W_out, n_tables]
        lookup_alt_indices     [B*H_out*W_out, n_tables, 3]
        lookup_alt_deltas      [B*H_out*W_out, n_tables, 3]
        lookup_indices_grad_c  [B*H_out*W_out, n_tables]         (zero carrier)
        lookup_alt_indices_grad_c [B*H_out*W_out, n_tables, 3]  (zero carrier)
    """

    @staticmethod
    def forward(ctx, x, anchors_a, anchors_b,
                B, C, H, W, H_out, W_out,
                kH, kW, sH, sW, pH, pW,
                cmp_eps, inv_l1):
        """
        Args:
            x:         [B, C, H, W] float32, contiguous, CUDA
            anchors_a: [n_tables, nap] int64
            anchors_b: [n_tables, nap] int64
        """
        native = _get_native_lutorch_manager()
        (
            lookup_indices,
            lookup_alt_indices,
            lookup_alt_deltas,
            anchor1_ids,
            anchor2_ids,
        ) = native.conv2d_anchor_pairs_lookup_na3_forward(
            x, anchors_a, anchors_b,
            H_out, W_out,
            kH, kW, sH, sW, pH, pW,
            float(cmp_eps),
        )

        n_patches = B * H_out * W_out
        n_tables  = anchors_a.shape[0]

        # Zero-value gradient carriers (same pattern as AnchorPairsLookupFunction)
        z = x.sum() * 0
        lookup_indices_grad_c     = z.expand(n_patches, n_tables)
        lookup_alt_indices_grad_c = z.expand(n_patches, n_tables, 3)

        ctx.inv_l1 = inv_l1
        ctx.B, ctx.C, ctx.H, ctx.W = B, C, H, W
        ctx.H_out, ctx.W_out = H_out, W_out
        ctx.kH, ctx.kW = kH, kW
        ctx.sH, ctx.sW = sH, sW
        ctx.pH, ctx.pW = pH, pW
        ctx.n_tables = n_tables
        ctx.save_for_backward(anchor1_ids, anchor2_ids, lookup_alt_deltas)

        return (
            lookup_indices,
            lookup_alt_indices,
            lookup_alt_deltas,
            lookup_indices_grad_c,
            lookup_alt_indices_grad_c,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            _,                             # grad for lookup_indices (not used here)
            _,                             # grad for lookup_alt_indices
            _,                             # grad for lookup_alt_deltas
            grad_main,                     # [B*H_out*W_out, n_tables]
            grad_alt,                      # [B*H_out*W_out, n_tables, 3]
        ) = grad_outputs

        anchor1_ids, anchor2_ids, lookup_alt_deltas = ctx.saved_tensors
        native = _get_native_lutorch_manager()

        grad_input = native.conv2d_anchor_pairs_lookup_na3_backward(
            anchor1_ids.reshape(-1).contiguous(),
            anchor2_ids.reshape(-1).contiguous(),
            lookup_alt_deltas.reshape(-1).contiguous(),
            grad_main.contiguous(),
            grad_alt.reshape(-1).contiguous(),
            ctx.B, ctx.C, ctx.H, ctx.W,
            ctx.H_out, ctx.W_out,
            ctx.kH, ctx.kW,
            ctx.sH, ctx.sW,
            ctx.pH, ctx.pW,
            ctx.n_tables,
            ctx.inv_l1,
        )

        # 17 inputs to forward → 17 gradients (None for non-tensor/int args)
        return (
            grad_input,  # x
            None,        # anchors_a
            None,        # anchors_b
            None, None, None, None, None, None,  # B,C,H,W,H_out,W_out
            None, None, None, None, None, None,  # kH,kW,sH,sW,pH,pW
            None,        # cmp_eps
            None,        # inv_l1
        )


# ---------------------------------------------------------------------------
# Conv2DLutFused module
# ---------------------------------------------------------------------------

class Conv2DLutFused(nn.Module):
    """
    Drop-in replacement for Conv2DLut that fuses spatial unfolding into
    the CUDA anchor-pairs lookup kernel (na3, smooth_mode=True only).

    Falls back to the reference Conv2DLut for any unsupported configuration
    (CPU, non-na3, non-smooth, or when native CUDA is unavailable).

    Args: identical to Conv2DLut.
    """

    def __init__(
        self,
        unfold_config: UnfoldConfiguration,
        in_channels: int,
        out_channels: int,
        n_anchor_pairs: int,
        n_heads: int = 1,
        tables_per_head: int = 1,
        device: Optional[torch.device] = None,
        **multi_head_lut_kwargs,
    ):
        super().__init__()

        if out_channels % n_heads != 0:
            raise ValueError(
                f"out_channels must be divisible by n_heads; "
                f"got out_channels={out_channels}, n_heads={n_heads}"
            )

        self.unfold_config  = unfold_config
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.n_heads        = n_heads
        self.tables_per_head = tables_per_head

        (kH, kW), (sH, sW), (pH, pW) = unfold_config.normalized()
        self.kH, self.kW = kH, kW
        self.sH, self.sW = sH, sW
        self.pH, self.pW = pH, pW

        H_p, W_p = unfold_config.output_spatial_shape()
        if H_p <= 0 or W_p <= 0:
            raise ValueError(
                f"Invalid patch grid from H={unfold_config.H}, W={unfold_config.W}, "
                f"kernel_size={unfold_config.kernel_size}, stride={unfold_config.stride}"
            )
        self.H_p, self.W_p = H_p, W_p
        self.n_patches = H_p * W_p

        patch_dim = in_channels * kH * kW
        self.patch_dim = patch_dim
        per_head_outputs = out_channels // n_heads
        self.per_head_outputs = per_head_outputs

        n_lookup_tables   = n_heads * tables_per_head
        n_entries_per_table = 2 ** n_anchor_pairs
        self.n_lookup_tables = n_lookup_tables
        self.n_anchor_pairs  = n_anchor_pairs

        # Pull out the parameters we need directly.
        n_alternatives  = int(multi_head_lut_kwargs.get("n_alternatives", 1))
        smooth_mode     = bool(multi_head_lut_kwargs.get("smooth_mode", False))
        cmp_eps         = float(multi_head_lut_kwargs.get("cmp_eps", 0.0))
        uncertainty_mode = multi_head_lut_kwargs.get(
            "uncertainty_mode", UncertaintyMode.INVERSE_L1
        )
        random_seed     = multi_head_lut_kwargs.get("random_seed", None)
        connected_anchors_mode = bool(multi_head_lut_kwargs.get("connected_anchors_mode", False))
        initial_weights_noise  = float(multi_head_lut_kwargs.get("initial_weights_noise", 0.001))

        self.n_alternatives  = n_alternatives
        self.smooth_mode     = smooth_mode
        self.cmp_eps         = cmp_eps
        self.uncertainty_mode = uncertainty_mode
        self.inv_l1          = (uncertainty_mode == UncertaintyMode.INVERSE_L1)

        # Determine if we can use the fused path.
        self._can_fuse = (
            n_alternatives == 3
            and smooth_mode
            and _USE_LUTORCH_CUSTOM_CUDA_KERNELS
            and _get_native_lutorch_manager() is not None
        )

        dev = device or torch.device("cpu")

        # Build anchor pairs (same as AnchorPairsLookup does internally).
        anchor_pairs_a, anchor_pairs_b = get_balanced_anchor_pairs(
            n_lookup_tables,
            n_anchor_pairs,
            patch_dim,
            dev,
            random_seed=random_seed,
            connected_mode=connected_anchors_mode,
            anchor_candidates=None,
        )
        self.register_buffer("anchor_pairs_a", anchor_pairs_a.contiguous())
        self.register_buffer("anchor_pairs_b", anchor_pairs_b.contiguous())

        # LProjection handles weight lookup and weight gradients.
        self.projection = LProjection(
            n_lookup_tables=n_lookup_tables,
            n_entries_per_table=n_entries_per_table,
            n_outputs=per_head_outputs,
            n_alternatives=n_alternatives,
            smooth_mode=smooth_mode,
            device=device,
            uncertainty_mode=uncertainty_mode,
        )
        if initial_weights_noise != 0.0:
            with torch.no_grad():
                rng_kwargs: dict = {"device": dev}
                if random_seed is not None:
                    rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed)
                self.projection.weights.add_(
                    torch.randn(self.projection.weights.shape, **rng_kwargs) * initial_weights_noise
                )

        # Fallback reference module — built only when fused path is unavailable.
        self._fallback: Optional[Conv2DLut] = None
        if not self._can_fuse:
            self._fallback = Conv2DLut(
                unfold_config=unfold_config,
                in_channels=in_channels,
                out_channels=out_channels,
                n_anchor_pairs=n_anchor_pairs,
                n_heads=n_heads,
                tables_per_head=tables_per_head,
                device=device,
                **multi_head_lut_kwargs,
            )

        # Cached table index tensors for LProjection.
        self._cached_table_indices_flat     = None
        self._cached_table_indices_alt_flat = None
        self._cached_n_patches              = None

    def _prepare_table_indices(self, n_patches: int, device: torch.device) -> None:
        if (
            self._cached_n_patches != n_patches
            or self._cached_table_indices_flat is None
            or self._cached_table_indices_flat.device != device
        ):
            table_idx = torch.arange(
                self.n_lookup_tables, dtype=torch.long, device=device
            ).view(1, self.n_lookup_tables)
            ti_expanded = table_idx.expand(n_patches, self.n_lookup_tables)
            self._cached_table_indices_flat = ti_expanded.flatten().contiguous()
            ti_alt = ti_expanded.unsqueeze(2).expand(
                n_patches, self.n_lookup_tables, self.n_alternatives
            )
            self._cached_table_indices_alt_flat = ti_alt.flatten().contiguous()
            self._cached_n_patches = n_patches

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        n_patches   = B * self.H_p * self.W_p

        self._prepare_table_indices(n_patches, x.device)
        self.projection._prepare_table_indices(n_patches, x.device)

        x_c = x.contiguous()

        if self.training:
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                lookup_indices_grad_c,
                lookup_alt_indices_grad_c,
            ) = Conv2DAnchorPairsLookupFunctionNa3.apply(
                x_c,
                self.anchor_pairs_a,
                self.anchor_pairs_b,
                B, C, H, W,
                self.H_p, self.W_p,
                self.kH, self.kW,
                self.sH, self.sW,
                self.pH, self.pW,
                self.cmp_eps,
                self.inv_l1,
            )
        else:
            # Eval: call CUDA kernel directly (no autograd), no grad carriers needed.
            native = _get_native_lutorch_manager()
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                _,
                _,
            ) = native.conv2d_anchor_pairs_lookup_na3_forward(
                x_c,
                self.anchor_pairs_a,
                self.anchor_pairs_b,
                self.H_p, self.W_p,
                self.kH, self.kW,
                self.sH, self.sW,
                self.pH, self.pW,
                float(self.cmp_eps),
            )
            lookup_indices_grad_c     = None
            lookup_alt_indices_grad_c = None

        # LProjection forward: [n_patches, n_lookup_tables, per_head_outputs]
        lut_out = self.projection(
            lookup_indices=lookup_indices,
            lookup_alt_indices=lookup_alt_indices,
            lookup_alt_deltas=lookup_alt_deltas,
            lookup_indices_grad_c=lookup_indices_grad_c,
            lookup_alt_indices_grad_c=lookup_alt_indices_grad_c,
        )

        # Reshape to [B, out_channels, H_p, W_p]
        # lut_out: [n_patches, n_lookup_tables, per_head_outputs]
        # = [B*H_p*W_p, n_heads*tables_per_head, per_head_outputs]
        total_outputs = self.n_heads * self.per_head_outputs  # == out_channels
        lut_out = lut_out.view(B * self.n_patches, self.n_heads, self.tables_per_head, self.per_head_outputs)
        lut_out = lut_out.sum(dim=2)  # [B*n_patches, n_heads, per_head_outputs]
        lut_out = lut_out.view(B * self.n_patches, total_outputs)
        lut_out = lut_out.view(B, self.n_patches, total_outputs)
        lut_out = lut_out.view(B, self.H_p, self.W_p, total_outputs)
        return lut_out.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels, H_p, W_p]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.shape}")

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"in_channels mismatch: expected {self.in_channels}, got {C}"
            )
        if (H, W) != (self.unfold_config.H, self.unfold_config.W):
            raise ValueError(
                f"Spatial size {(H, W)} does not match config "
                f"({self.unfold_config.H}, {self.unfold_config.W})"
            )

        use_fused = (
            self._can_fuse
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float64)
        )
        if use_fused:
            return self._forward_fused(x)

        # Fallback: build a reference Conv2DLut on-the-fly if needed.
        if self._fallback is None:
            # Create fallback sharing the same weights.
            raise RuntimeError(
                "Conv2DLutFused: fused path unavailable and no fallback was built. "
                "This should not happen — please report a bug."
            )
        return self._fallback(x)
