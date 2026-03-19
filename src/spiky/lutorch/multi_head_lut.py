"""
Multi-head lookup table module combining AnchorPairsLookup and LProjection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Union

from spiky.lutorch.anchor_pairs_lookup import AnchorPairsLookup
from spiky.lutorch.l_projection import LProjection
from spiky.lutorch.lut_helpers import UncertaintyMode


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
        connected_anchors_mode: If True, anchor pairs form a connected graph (default: False)
        anchor_candidates: Optional tensor of shape [tables_per_head, n_heads, max_anchors_per_table]
                          with input indices. If None, balanced coverage over input dims is used.
        cmp_eps: Epsilon for comparison (default: 0.0)
        random_seed: Random seed for anchor pair sampling
        n_alternatives: Number of alternative lookup indices per table (default: 1)
        smooth_mode: If True, use smooth interpolation in LProjection (default: False)
        device: Device to place buffers on
        initial_weights_noise: Std of zero-mean Gaussian added to projection weights at init (default: 0.0).
    """
    
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        n_buckets: int = 1,
        connected_anchors_mode: bool = False,
        anchor_candidates: Optional[torch.Tensor] = None,
        cmp_eps: float = 0.0,
        random_seed: Optional[int] = None,
        n_alternatives: int = 1,
        smooth_mode: bool = False,
        device: Optional[torch.device] = None,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
        initial_weights_noise: float = 0.001,
        table_dropout: float = 0.0,
        dropout: float = 0.0,
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
        self.table_dropout = table_dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Total number of lookup tables
        n_lookup_tables = n_heads * tables_per_head
        
        # n_entries_per_table is derived from n_anchor_pairs, multiplied by n_buckets
        n_entries_per_table = (2 ** n_anchor_pairs) * n_buckets
        
        # Reshape anchor_candidates when provided: [tables_per_head, n_heads, max_anchors_per_table]
        # -> [n_heads * tables_per_head, max_anchors_per_table] for AnchorPairsLookup.
        reshaped_anchor_candidates = None
        if anchor_candidates is not None:
            if not isinstance(anchor_candidates, torch.Tensor) or anchor_candidates.dim() != 3:
                raise ValueError(
                    "anchor_candidates must be a 3D tensor of shape "
                    "[tables_per_head, n_heads, max_anchors_per_table], or None"
                )
            if anchor_candidates.shape[0] != tables_per_head or anchor_candidates.shape[1] != n_heads:
                raise ValueError(
                    "anchor_candidates tensor must have shape "
                    f"[tables_per_head={tables_per_head}, n_heads={n_heads}, max_anchors_per_table], "
                    f"got {tuple(anchor_candidates.shape)}"
                )
            reshaped_anchor_candidates = (
                anchor_candidates.permute(1, 0, 2).contiguous().view(n_lookup_tables, -1)
            )
        
        # Create AnchorPairsLookup: n_lookup_tables total
        self.lookup = AnchorPairsLookup(
            input_dim=input_dim,
            n_tables=n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            connected_anchors_mode=connected_anchors_mode,
            anchor_candidates=reshaped_anchor_candidates,
            cmp_eps=cmp_eps,
            random_seed=random_seed,
            device=device,
            n_alternatives=n_alternatives,
            uncertainty_mode=uncertainty_mode,
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
        if initial_weights_noise != 0.0:
            dev = device or torch.device("cpu")
            with torch.no_grad():
                rng_kwargs: dict = {"device": dev}
                if random_seed is not None:
                    rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed)
                self.projection.weights.add_(
                    torch.randn(self.projection.weights.shape, **rng_kwargs) * initial_weights_noise
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
        # Apply table dropout: randomly zero entire tables during training
        if self.training and self.table_dropout > 0.0:
            mask = torch.bernoulli(
                torch.full((output.shape[0], self.n_heads, self.tables_per_head, 1),
                           1.0 - self.table_dropout, device=output.device)
            )
            output = output * mask / (1.0 - self.table_dropout)
        if self.dropout is not None:
            output = self.dropout(output)
        # Sum over tables_per_head dimension
        output = output.sum(dim=2)  # [B, n_heads, n_outputs]
        
        return output


@dataclass(frozen=True)
class UnfoldConfiguration:
    """
    Configuration of 2D unfold-style patches.

    All spatial parameters mirror a simplified conv2d / unfold API (dilation fixed to 1).
    Each field can be either an int (applied to both dimensions) or a (height, width) tuple.
    """

    H: int
    W: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = 1
    padding: Union[int, Tuple[int, int]] = 0

    @staticmethod
    def _to_2d(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(v, tuple):
            if len(v) != 2:
                raise ValueError(f"Expected a tuple of length 2, got {v}")
            return v
        return v, v

    def normalized(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Returns:
            (kernel_size_2d, stride_2d, padding_2d), each as (H, W) tuples.
        """
        kH, kW = self._to_2d(self.kernel_size)
        sH, sW = self._to_2d(self.stride)
        pH, pW = self._to_2d(self.padding)
        return (kH, kW), (sH, sW), (pH, pW)

    def output_spatial_shape(self) -> Tuple[int, int]:
        """
        Compute the unfolded patch grid size (H_p, W_p) for an input of size (H, W).
        Uses same formula as conv2d with dilation=1: (in + 2*pad - kernel) // stride + 1.
        """
        (kH, kW), (sH, sW), (pH, pW) = self.normalized()

        def _out_dim(in_size: int, kernel: int, stride: int, pad: int) -> int:
            return (in_size + 2 * pad - kernel) // stride + 1

        H_p = _out_dim(self.H, kH, sH, pH)
        W_p = _out_dim(self.W, kW, sW, pW)
        return H_p, W_p


class ProjectionLUT(nn.Module):
    """
    Projection LUT built on top of ``MultiHeadLut`` using a flat 2D input and
    unfold-style patching.

    This module:
      - Takes a flat spatial input of shape ``[B, H, W]``.
      - Defines patches using unfold parameters (kernel_size / stride /
        padding / dilation).
      - Builds a ``MultiHeadLut`` with:
          * ``input_dim = H * W`` (flattened spatial grid)
          * ``n_heads = n_patches`` (one head per patch)
          * ``n_outputs = n_outputs``
        and per-table ``anchor_candidates`` restricted to the indices of the
        corresponding patch.
      - In forward, applies this ``MultiHeadLut`` to the flattened input and
        reshapes the result to ``[B, H_p, W_p, O]`` where ``H_p, W_p`` are the
        number of patches along height and width.
    """

    def __init__(
        self,
        unfold_config: UnfoldConfiguration,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        fold_config: Optional[UnfoldConfiguration] = None,
        device: Optional[torch.device] = None,
        **multi_head_lut_kwargs,
    ):
        super().__init__()

        self.unfold_config = unfold_config
        self.fold_config = fold_config
        self.n_outputs = n_outputs

        (kH, kW), (sH, sW), (pH, pW) = self.unfold_config.normalized()
        if (pH, pW) != (0, 0):
            raise ValueError(
                f"ProjectionLUT requires unfold_config padding to be 0; got padding=({pH}, {pW})"
            )

        # Compute patch grid size as in conv2d / unfold (zero padding, dilation=1)
        H_p, W_p = self.unfold_config.output_spatial_shape()
        if H_p <= 0 or W_p <= 0:
            raise ValueError(
                f"Invalid patch grid size computed from H={self.unfold_config.H}, W={self.unfold_config.W}, "
                f"kernel_size={self.unfold_config.kernel_size}, "
                f"stride={self.unfold_config.stride}"
            )

        self.H_p = H_p
        self.W_p = W_p
        self.n_patches = H_p * W_p

        input_dim = self.unfold_config.H * self.unfold_config.W
        n_heads = self.n_patches

        # Build per-head candidate indices corresponding to each patch window
        # using vectorized unfold over a grid of flat indices.
        dev = device or torch.device("cpu")
        index_grid = torch.arange(
            input_dim,
            device=dev,
            dtype=torch.long,
        ).view(1, 1, self.unfold_config.H, self.unfold_config.W)

        # Unfold directly (zero padding, dilation=1).
        # Result shape from unfold: [1, K, n_patches] where K = kH * kW
        patches = F.unfold(
            index_grid.to(dtype=torch.float32),
            kernel_size=(kH, kW),
            dilation=1,
            stride=(sH, sW),
        )
        # After transpose: [1, n_patches, K]
        patches = patches.to(dtype=torch.long).transpose(1, 2).contiguous()

        # anchor_candidates: [tables_per_head, n_heads, K] (all indices are valid and shared across tables in a head)
        # Repeat along the batch dimension to create one candidate set per table_per_head.
        anchor_candidates = patches.repeat(tables_per_head, 1, 1).to(device=dev)

        # Optional folding configuration: map per-patch outputs back to an
        # output spatial grid using scatter-add.
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
                self.H_out * self.W_out,
                device=dev,
                dtype=torch.long,
            ).view(1, 1, self.H_out, self.W_out)

            patches_out = F.unfold(
                index_grid_out.to(dtype=torch.float32),
                kernel_size=(kH_f, kW_f),
                dilation=1,
                stride=(sH_f, sW_f),
            )  # [1, K_f, n_patches]
            # Squeeze batch dimension to get [K_f, n_patches], then transpose to [n_patches, K_f]
            patches_out = (
                patches_out.squeeze(0)
                .to(dtype=torch.long)
                .transpose(0, 1)
                .contiguous()
            )  # [n_patches, K_f]

            K_f = patches_out.shape[1]
            if K_f < self.n_outputs:
                raise ValueError(
                    f"Each fold patch must contain at least O indices; got K_f={K_f} < O={self.O}"
                )

            # Select n_outputs random, unique kernel positions (columns) per patch in a
            # fully vectorised way. We generate random scores per (patch, pos)
            # and take top-n_outputs columns for each patch.
            rnd = torch.rand(self.n_patches, K_f, device=dev)
            selected_cols = torch.topk(rnd, k=self.n_outputs, dim=1, largest=True).indices  # [n_patches, O]

            # Map to flat output indices for each patch: [n_patches, O]
            fold_output_indices = patches_out.gather(1, selected_cols).contiguous()
            self.register_buffer("fold_output_indices", fold_output_indices)
            # Also store a flattened, per-position base index vector for efficient
            # batched scatter-add in forward.
            self.register_buffer(
                "fold_output_indices_flat",
                fold_output_indices.view(1, -1),
            )
            # Cache for batch offsets used in flattened scatter_add; recomputed
            # when batch size or device changes.
            self.register_buffer("_cached_batch_offsets", None)

        # Construct MultiHeadLut; we control input_dim / n_heads / n_outputs here.
        self.lut = MultiHeadLut(
            input_dim=input_dim,
            n_heads=n_heads,
            n_outputs=self.n_outputs,
            n_anchor_pairs=n_anchor_pairs,
            tables_per_head=tables_per_head,
            anchor_candidates=anchor_candidates,
            device=device,
            **multi_head_lut_kwargs,
        )
        self._cached_batch_offsets = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, H, W] where (H, W) match
               ``unfold_config.H`` and ``unfold_config.W``.

        Returns:
            - If ``fold_config`` is None: tensor of shape [B, H_p, W_p, O].
            - If ``fold_config`` is set: tensor of shape [B, H_out, W_out].
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, H, W], got shape {x.shape}")

        B, H, W = x.shape
        if (H, W) != (self.unfold_config.H, self.unfold_config.W):
            raise ValueError(
                f"Input spatial size {(H, W)} does not match ProjectionLUT configuration "
                f"({self.unfold_config.H}, {self.unfold_config.W})"
            )

        x_flat = x.view(B, H * W)
        lut_out = self.lut(x_flat)  # [B, n_patches, n_outputs]

        if self.fold_config is None:
            # Return per-patch outputs.
            return lut_out.view(B, self.H_p, self.W_p, self.n_outputs)

        B, P, O = lut_out.shape  # P == n_patches, O == n_outputs
        flat_out_dim = self.H_out * self.W_out

        # Flattened 1D scatter_add over all batches for robust gradient flow.
        # fold_output_indices: [P, O] with flat spatial indices.
        indices = self.fold_output_indices  # [P, O]
        indices_exp = indices.unsqueeze(0).expand(B, -1, -1)  # [B, P, O]
        values = lut_out  # [B, P, O]

        indices_flat = indices_exp.reshape(-1)  # [B * P * O]
        values_flat = values.reshape(-1)        # [B * P * O]

        expected_len = B * P * O
        if (
            self._cached_batch_offsets is None
            or self._cached_batch_offsets.numel() != expected_len
            or self._cached_batch_offsets.device != lut_out.device
        ):
            self._cached_batch_offsets = (
                torch.arange(B, device=lut_out.device, dtype=torch.long)
                .repeat_interleave(P * O) * flat_out_dim
            )

        scatter_indices = indices_flat + self._cached_batch_offsets  # [B * P * O]

        out_flat = torch.zeros(
            B * flat_out_dim,
            device=lut_out.device,
            dtype=lut_out.dtype,
        )
        out_flat.scatter_add_(0, scatter_indices, values_flat)

        return out_flat.view(B, self.H_out, self.W_out)


class Conv2DLut(nn.Module):
    """
    2D convolution-style LUT built on top of ``MultiHeadLut`` for inputs
    of shape ``[B, C, H, W]``.

    This module:
      - Takes an input of shape ``[B, C, H, W]``.
      - Uses an ``UnfoldConfiguration`` (same as ``ProjectionLUT``) to define
        spatial patches over ``(H, W)`` with kernel / stride.
      - Computes ``n_patches = H_p * W_p`` where ``(H_p, W_p)`` is the unfolded
        patch grid.
      - For each patch, constructs a flattened vector of size
        ``patch_dim = C * kH * kW`` using ``torch.nn.functional.unfold``.
      - Builds a ``MultiHeadLut`` with:
          * ``input_dim = patch_dim``
          * ``n_heads = n_heads`` (typically small, default 1)
          * ``n_outputs = out_channels // n_heads``
      - In forward:
          * unfold the input into patches
          * Reshapes patches to ``[B * n_patches, patch_dim]``.
          * Applies the internal ``MultiHeadLut`` to obtain
            ``[B * n_patches, n_heads, out_channels // n_heads]``.
          * Reshapes to ``[B, out_channels, H_p, W_p]`` as the final output.
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
                f"out_channels must be divisible by n_heads; got "
                f"out_channels={out_channels}, n_heads={n_heads}"
            )

        self.unfold_config = unfold_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads

        (kH, kW), _, _ = self.unfold_config.normalized()

        # Compute patch grid size (supports padding).
        H_p, W_p = self.unfold_config.output_spatial_shape()
        if H_p <= 0 or W_p <= 0:
            raise ValueError(
                f"Invalid patch grid size computed from H={self.unfold_config.H}, W={self.unfold_config.W}, "
                f"kernel_size={self.unfold_config.kernel_size}, "
                f"stride={self.unfold_config.stride}"
            )

        self.H_p = H_p
        self.W_p = W_p
        self.n_patches = H_p * W_p

        # Each patch is C * kH * kW features.
        patch_dim = in_channels * kH * kW
        self.patch_dim = patch_dim

        # MultiHeadLut operates on individual patch vectors.
        per_head_outputs = out_channels // n_heads

        self.lut = MultiHeadLut(
            input_dim=patch_dim,
            n_heads=n_heads,
            n_outputs=per_head_outputs,
            n_anchor_pairs=n_anchor_pairs,
            tables_per_head=tables_per_head,
            device=device,
            **multi_head_lut_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W] where (H, W) match
               ``unfold_config.H`` and ``unfold_config.W``.

        Returns:
            Tensor of shape [B, out_channels, H_p, W_p].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got shape {x.shape}")

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Input channels C={C} do not match Conv2DLut configuration "
                f"in_channels={self.in_channels}"
            )
        if (H, W) != (self.unfold_config.H, self.unfold_config.W):
            raise ValueError(
                f"Input spatial size {(H, W)} does not match Conv2DLut configuration "
                f"({self.unfold_config.H}, {self.unfold_config.W})"
            )

        (kH, kW), (sH, sW), (pH, pW) = self.unfold_config.normalized()

        # Unfold input into patches: [B, C * kH * kW, n_patches]
        patches = F.unfold(
            x,
            kernel_size=(kH, kW),
            dilation=1,
            padding=(pH, pW),
            stride=(sH, sW),
        )  # [B, C * kH * kW, n_patches]

        # Rearrange to [B * n_patches, patch_dim]
        patches = patches.transpose(1, 2).contiguous()  # [B, n_patches, patch_dim]
        patches_flat = patches.view(B * self.n_patches, self.patch_dim)

        # Apply MultiHeadLut: [B * n_patches, n_heads, out_channels // n_heads]
        lut_out = self.lut(patches_flat)
        BnP, n_heads, per_head_outputs = lut_out.shape
        total_outputs = n_heads * per_head_outputs
        # [B * n_patches, n_heads, per_head_outputs] -> [B * n_patches, out_channels]
        lut_out_flat = lut_out.view(BnP, total_outputs)

        # Reshape to [B, n_patches, out_channels] then [B, out_channels, H_p, W_p]
        lut_out_patches = lut_out_flat.view(B, self.n_patches, total_outputs)
        lut_out_spatial = lut_out_patches.view(B, self.H_p, self.W_p, total_outputs)
        return lut_out_spatial.permute(0, 3, 1, 2).contiguous()
