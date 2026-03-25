"""
WTA module combining WTALookup and LProjection, and spatial wrappers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from spiky.lutorch.wta_lookup import WTALookup
from spiky.lutorch.l_projection import LProjection
from spiky.lutorch.lut_helpers import UncertaintyMode
from spiky.lutorch.multi_head_lut import UnfoldConfiguration


class WTA(nn.Module):
    """
    WTA (Winner-Take-All) module combining WTALookup and LProjection.

    Processes input of shape [B, n_channels, n_inputs]. Each channel runs an
    independent WTA (argmax + n_alternatives runner-ups) over its n_inputs
    values. LProjection maps the selected indices to output features.

    Args:
        n_channels: Number of independent WTA channels (C in [B, C, n_inputs]).
        n_inputs: Number of candidates per channel
        n_outputs: Number of output dimensions per channel.
        n_alternatives: Number of runner-up indices per channel (default: 1).
        smooth_mode: If True, use smooth interpolation in LProjection (default: False).
        device: Device to place weight buffers on.
        uncertainty_mode: Uncertainty function for gradient weighting.
        initial_weights_noise: Std of Gaussian added to LProjection weights at init.
        dropout: Dropout probability applied to the output (default: 0.0).
        random_seed: Seed for weight initialisation noise.
        normalize_weights: Passed to LProjection: L2 column-normalize projection weights at the start
            of each forward while training (see :class:`~spiky.lutorch.l_projection.LProjection`).
    """

    def __init__(
        self,
        n_channels: int,
        n_inputs: int,
        n_outputs: int,
        n_alternatives: int = 1,
        smooth_mode: bool = False,
        device: Optional[torch.device] = None,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
        initial_weights_noise: float = 0.001,
        dropout: float = 0.0,
        random_seed: Optional[int] = None,
        normalize_weights: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_alternatives = n_alternatives
        self.smooth_mode = smooth_mode
        self.uncertainty_mode = uncertainty_mode
        self.normalize_weights = normalize_weights
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.wta_lookup = WTALookup(n_inputs, n_alternatives, uncertainty_mode)
        self.projection = LProjection(
            n_lookup_tables=n_channels,
            n_entries_per_table=n_inputs,
            n_outputs=n_outputs,
            n_alternatives=n_alternatives,
            smooth_mode=smooth_mode,
            device=device,
            uncertainty_mode=uncertainty_mode,
            normalize_weights=normalize_weights,
        )
        if initial_weights_noise != 0.0:
            dev = device or torch.device("cpu")
            with torch.no_grad():
                rng_kwargs: dict = {"device": dev}
                if random_seed is not None:
                    rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed)
                self.projection.weights.add_(
                    torch.randn(self.projection.weights.shape, **rng_kwargs) * initial_weights_noise
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, n_channels, n_inputs].

        Returns:
            Output tensor of shape [B, n_channels, n_outputs].
        """
        if self.training:
            (
                lookup_indices,
                lookup_alt_indices,
                lookup_alt_deltas,
                lookup_indices_grad_c,
                lookup_alt_indices_grad_c,
            ) = self.wta_lookup(x)
        else:
            lookup_indices, lookup_alt_indices, lookup_alt_deltas = self.wta_lookup(x)
            lookup_indices_grad_c = None
            lookup_alt_indices_grad_c = None

        output = self.projection(
            lookup_indices=lookup_indices,
            lookup_alt_indices=lookup_alt_indices,
            lookup_alt_deltas=lookup_alt_deltas,
            lookup_indices_grad_c=lookup_indices_grad_c,
            lookup_alt_indices_grad_c=lookup_alt_indices_grad_c,
        )  # [B, n_channels, n_outputs]

        if self.dropout is not None:
            output = self.dropout(output)

        return output


class ProjectionWTA(nn.Module):
    """
    Projection WTA built on top of WTA using 2D unfold-style patching.

    Takes a flat spatial input of shape [B, H, W], unfolds it into patches of
    size kH * kW, and applies WTA independently to each patch. Each patch
    position becomes one WTA channel; the WTA selects the strongest value
    within the kernel window and maps it to n_outputs features.

    Without fold_config, output shape is [B, H_p, W_p, n_outputs].

    With fold_config, the per-patch outputs are scattered back into an output
    spatial grid via scatter-add, producing shape [B, H_out, W_out]. This
    mirrors the fold_config behaviour of ProjectionLUT.

    Args:
        unfold_config: UnfoldConfiguration describing input shape and patch
            geometry. Padding must be 0.
        n_outputs: Number of output dimensions per patch.
        fold_config: Optional UnfoldConfiguration for the output grid. Must
            produce the same number of patches as unfold_config.
        device: Device for weight buffers.
        **wta_kwargs: Forwarded to WTA (e.g. n_alternatives, smooth_mode, normalize_weights, etc.).
    """

    def __init__(
        self,
        unfold_config: UnfoldConfiguration,
        n_outputs: int,
        fold_config: Optional[UnfoldConfiguration] = None,
        device: Optional[torch.device] = None,
        **wta_kwargs,
    ):
        super().__init__()
        self.unfold_config = unfold_config
        self.fold_config = fold_config
        self.n_outputs = n_outputs

        (kH, kW), (sH, sW), (pH, pW) = self.unfold_config.normalized()
        if (pH, pW) != (0, 0):
            raise ValueError(
                f"ProjectionWTA requires unfold_config padding to be 0; got padding=({pH}, {pW})"
            )

        H_p, W_p = self.unfold_config.output_spatial_shape()
        if H_p <= 0 or W_p <= 0:
            raise ValueError(
                f"Invalid patch grid size from H={self.unfold_config.H}, W={self.unfold_config.W}, "
                f"kernel_size={self.unfold_config.kernel_size}, stride={self.unfold_config.stride}"
            )

        self.H_p = H_p
        self.W_p = W_p
        self.n_patches = H_p * W_p
        self._kH = kH
        self._kW = kW
        self._sH = sH
        self._sW = sW

        dev = device or torch.device("cpu")

        if self.fold_config is not None:
            (kH_f, kW_f), (sH_f, sW_f), _ = self.fold_config.normalized()
            H_p_f, W_p_f = self.fold_config.output_spatial_shape()
            n_patches_f = H_p_f * W_p_f
            if n_patches_f != self.n_patches:
                raise ValueError(
                    f"fold_config must produce the same number of patches as unfold_config "
                    f"({n_patches_f} != {self.n_patches})"
                )

            self.H_out = self.fold_config.H
            self.W_out = self.fold_config.W

            index_grid_out = torch.arange(
                self.H_out * self.W_out, device=dev, dtype=torch.long,
            ).view(1, 1, self.H_out, self.W_out)

            patches_out = F.unfold(
                index_grid_out.to(dtype=torch.float32),
                kernel_size=(kH_f, kW_f),
                dilation=1,
                stride=(sH_f, sW_f),
            )  # [1, K_f, n_patches]
            patches_out = (
                patches_out.squeeze(0).to(dtype=torch.long).transpose(0, 1).contiguous()
            )  # [n_patches, K_f]

            K_f = patches_out.shape[1]
            if K_f < self.n_outputs:
                raise ValueError(
                    f"Each fold patch must contain at least n_outputs indices; "
                    f"got K_f={K_f} < n_outputs={self.n_outputs}"
                )

            rnd = torch.rand(self.n_patches, K_f, device=dev)
            selected_cols = torch.topk(rnd, k=self.n_outputs, dim=1, largest=True).indices
            fold_output_indices = patches_out.gather(1, selected_cols).contiguous()
            self.register_buffer("fold_output_indices", fold_output_indices)
            self.register_buffer("fold_output_indices_flat", fold_output_indices.view(1, -1))
            self.register_buffer("_cached_batch_offsets", None)

        self.wta = WTA(
            n_channels=self.n_patches,
            n_inputs=kH * kW,
            n_outputs=n_outputs,
            device=device,
            **wta_kwargs,
        )
        self._cached_batch_offsets = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, H, W].

        Returns:
            - If fold_config is None: tensor of shape [B, H_p, W_p, n_outputs].
            - If fold_config is set: tensor of shape [B, H_out, W_out].
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, H, W], got shape {x.shape}")

        B, H, W = x.shape
        if (H, W) != (self.unfold_config.H, self.unfold_config.W):
            raise ValueError(
                f"Input spatial size {(H, W)} does not match ProjectionWTA configuration "
                f"({self.unfold_config.H}, {self.unfold_config.W})"
            )

        # [B, kH*kW, n_patches] -> [B, n_patches, kH*kW]
        patches = F.unfold(
            x.unsqueeze(1),
            kernel_size=(self._kH, self._kW),
            dilation=1,
            padding=0,
            stride=(self._sH, self._sW),
        ).transpose(1, 2).contiguous()

        # [B, n_patches, n_outputs]
        out = self.wta(patches)

        if self.fold_config is None:
            return out.view(B, self.H_p, self.W_p, self.n_outputs)

        P, O = self.n_patches, self.n_outputs
        flat_out_dim = self.H_out * self.W_out

        indices_exp = self.fold_output_indices.unsqueeze(0).expand(B, -1, -1)  # [B, P, O]
        indices_flat = indices_exp.reshape(-1)
        values_flat = out.reshape(-1)

        expected_len = B * P * O
        if (
            self._cached_batch_offsets is None
            or self._cached_batch_offsets.numel() != expected_len
            or self._cached_batch_offsets.device != out.device
        ):
            self._cached_batch_offsets = (
                torch.arange(B, device=out.device, dtype=torch.long)
                .repeat_interleave(P * O) * flat_out_dim
            )

        scatter_indices = indices_flat + self._cached_batch_offsets

        out_flat = torch.zeros(B * flat_out_dim, device=out.device, dtype=out.dtype)
        out_flat.scatter_add_(0, scatter_indices, values_flat)

        return out_flat.view(B, self.H_out, self.W_out)


class Conv2DWTA(nn.Module):
    """
    2D convolution-style WTA built on top of WTA for inputs of shape [B, C, H, W].

    Unfolds the input into spatial patches of size C * kH * kW. Each patch is
    treated as a single WTA channel: WTA selects the winning feature position
    within the flattened patch and maps it to out_channels output features via
    LProjection. Weights are shared across all spatial locations.

    Output shape: [B, out_channels, H_p, W_p].

    Args:
        unfold_config: UnfoldConfiguration describing input shape and patch geometry.
        in_channels: Number of input channels C.
        out_channels: Number of output channels.
        device: Device for weight buffers.
        **wta_kwargs: Forwarded to WTA (including normalize_weights).
    """

    def __init__(
        self,
        unfold_config: UnfoldConfiguration,
        in_channels: int,
        out_channels: int,
        device: Optional[torch.device] = None,
        **wta_kwargs,
    ):
        super().__init__()

        self.unfold_config = unfold_config
        self.in_channels = in_channels
        self.out_channels = out_channels

        (kH, kW), _, _ = self.unfold_config.normalized()

        H_p, W_p = self.unfold_config.output_spatial_shape()
        if H_p <= 0 or W_p <= 0:
            raise ValueError(
                f"Invalid patch grid size from H={self.unfold_config.H}, W={self.unfold_config.W}, "
                f"kernel_size={self.unfold_config.kernel_size}, stride={self.unfold_config.stride}"
            )

        self.H_p = H_p
        self.W_p = W_p
        self.n_patches = H_p * W_p
        self.patch_dim = in_channels * kH * kW

        # One WTA channel per spatial location; weights shared across locations.
        self.wta = WTA(
            n_channels=1,
            n_inputs=self.patch_dim,
            n_outputs=out_channels,
            device=device,
            **wta_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Tensor of shape [B, out_channels, H_p, W_p].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got shape {x.shape}")

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Input channels C={C} do not match Conv2DWTA in_channels={self.in_channels}"
            )
        if (H, W) != (self.unfold_config.H, self.unfold_config.W):
            raise ValueError(
                f"Input spatial size {(H, W)} does not match Conv2DWTA configuration "
                f"({self.unfold_config.H}, {self.unfold_config.W})"
            )

        (kH, kW), (sH, sW), (pH, pW) = self.unfold_config.normalized()

        # [B, patch_dim, n_patches] -> [B*n_patches, 1, patch_dim]
        patches = F.unfold(
            x,
            kernel_size=(kH, kW),
            dilation=1,
            padding=(pH, pW),
            stride=(sH, sW),
        ).transpose(1, 2).contiguous().view(B * self.n_patches, 1, self.patch_dim)

        # [B*n_patches, 1, out_channels] -> [B, out_channels, H_p, W_p]
        wta_out = self.wta(patches)
        return (
            wta_out
            .view(B, self.n_patches, self.out_channels)
            .view(B, self.H_p, self.W_p, self.out_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
