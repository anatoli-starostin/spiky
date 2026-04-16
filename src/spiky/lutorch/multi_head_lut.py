"""
Multi-head lookup table module combining AnchorPairsLookup and LProjection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Union

# When True, all MultiHeadLut instances use recompute_in_backward regardless of constructor arg.
# Set by monkeypatching in tests (e.g. via the autouse fixture in conftest.py).
_FORCE_RECOMPUTE_IN_BACKWARD = False

from spiky.lutorch.anchor_pairs_lookup import AnchorPairsLookup, _compute_anchor_data, _anchor_pairs_backward
from spiky.lutorch.l_projection import LProjection, _lprojection_backward, _forward_smooth_impl, _get_native_lutorch_manager as _lp_native, _USE_LUTORCH_CUSTOM_CUDA_KERNELS as _LP_CUDA, _LUTORCH_CUDA_THREADS_PER_BLOCK as _LP_TPB
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy, compute_hierarchical_n_tables, compute_multiscale_n_tables, compute_conv2d_n_tables


class MultiHeadLutFunction(torch.autograd.Function):
    """
    Fused autograd function for AnchorPairsLookup + LProjection.

    Saves only (x, weights, anchor_pairs_a, anchor_pairs_b, powers, main_weight, alt_weight).
    Recomputes lookup_indices / anchor ids in backward from the small anchor buffers,
    avoiding ~264 MB of saved tensors per block.

    Non-smooth mode uses F.embedding_bag (lookup + sum fused) to avoid materialising
    the [B, n_tables, n_outputs] intermediate (~2 GB) during forward.
    Returns output already summed over tables_per_head: [B, n_heads, n_outputs].
    Smooth mode returns [B, n_tables, n_outputs] as before (requires per-table weights).
    """

    @staticmethod
    def forward(ctx, x, weights, anchor_pairs_a, anchor_pairs_b, powers,
                cmp_eps, n_alternatives, smooth_mode, uncertainty_mode_int,
                uncertainty_bias, batch_offset):
        lookup_indices, lookup_alt_indices, lookup_alt_deltas, _, _ = \
            _compute_anchor_data(x, anchor_pairs_a, anchor_pairs_b, powers, cmp_eps, n_alternatives)

        batch_size = x.shape[0]
        n_tables = weights.shape[0]
        l1_uncertainty = (uncertainty_mode_int == 0)

        table_indices_flat = torch.arange(n_tables, device=x.device, dtype=torch.long).repeat(batch_size)
        table_indices_alt_flat = (
            torch.arange(n_tables, device=x.device, dtype=torch.long)
            .unsqueeze(1).expand(-1, n_alternatives).reshape(-1).repeat(batch_size)
        )
        table_indices_expanded = table_indices_flat.view(batch_size, n_tables)
        table_indices_expanded_alt = table_indices_alt_flat.view(batch_size, n_tables, n_alternatives)

        if not smooth_mode:
            output = weights[table_indices_expanded, lookup_indices]
            main_weight = alt_weight = None
        else:
            use_native = (
                _LP_CUDA and _lp_native() is not None
                and weights.is_cuda and weights.dtype in (torch.float32, torch.float64)
                and n_alternatives <= 3
            )
            if use_native:
                output, main_weight, alt_weight = _lp_native().lprojection_forward_smooth(
                    weights,
                    lookup_indices.contiguous(), lookup_alt_indices.contiguous(),
                    lookup_alt_deltas.contiguous(),
                    table_indices_flat.contiguous(), table_indices_alt_flat.contiguous(),
                    l1_uncertainty, uncertainty_bias, _LP_TPB,
                )
            else:
                output, main_weight, alt_weight = _forward_smooth_impl(
                    weights, table_indices_expanded, table_indices_expanded_alt,
                    lookup_indices, lookup_alt_indices, lookup_alt_deltas,
                    n_alternatives, l1_uncertainty, uncertainty_bias,
                )

        ctx.save_for_backward(x, weights, anchor_pairs_a, anchor_pairs_b, powers,
                              main_weight, alt_weight)
        ctx.cmp_eps = cmp_eps
        ctx.n_alternatives = n_alternatives
        ctx.smooth_mode = smooth_mode
        ctx.inv_l1 = l1_uncertainty
        ctx.uncertainty_bias = float(uncertainty_bias)
        ctx.batch_offset = batch_offset

        return output, lookup_alt_deltas

    @staticmethod
    def backward(ctx, grad_output, grad_lookup_alt_deltas):
        x, weights, anchor_pairs_a, anchor_pairs_b, powers, main_weight, alt_weight = ctx.saved_tensors

        # Ensure grad_output matches weights dtype for lprojection backward,
        # and fp32 for anchor backward (which needs to match x dtype)
        if grad_output.dtype != weights.dtype:
            grad_output = grad_output.to(weights.dtype)

        with torch.no_grad():
            lookup_indices, lookup_alt_indices, lookup_alt_deltas, anchor1_ids, anchor2_ids = \
                _compute_anchor_data(x, anchor_pairs_a, anchor_pairs_b, powers,
                                     ctx.cmp_eps, ctx.n_alternatives)

        weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad = \
            _lprojection_backward(
                grad_output, weights, lookup_indices, lookup_alt_indices,
                main_weight, alt_weight, ctx.smooth_mode, ctx.n_alternatives,
            )

        # anchor backward needs fp32 (same dtype as x)
        x_grad_flat = _anchor_pairs_backward(
            x, anchor1_ids, anchor2_ids, lookup_alt_deltas,
            ctx.inv_l1, ctx.batch_offset, ctx.uncertainty_bias,
            lookup_indices_grad_c_grad.float(), lookup_alt_indices_grad_c_grad.float(),
            grad_lookup_alt_deltas,
        )

        # 11 inputs -> 11 gradient returns
        return (x_grad_flat.view(x.shape), weights_grad,
                None, None, None, None, None, None, None, None, None)


def _calibrate_per_head_output(output: torch.Tensor) -> torch.Tensor:
    """
    For each batch row and each head, scale that row's outputs by (max − min) over ``n_outputs`` only.

    Args:
        output: [B, n_heads, n_outputs]
    """
    vmin = output.min(dim=-1, keepdim=True)[0]
    vmax = output.max(dim=-1, keepdim=True)[0]
    scale = (vmax - vmin).clamp_min(1e-20)
    return output / scale


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
        normalize_weights: Passed to LProjection: L2 column-normalize weights after each backward.
        calibrate_output: If True, per (batch, head) divide by (max − min) over ``n_outputs`` only (after dropout and sum).
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
        normalize_weights: bool = False,
        calibrate_output: bool = False,
        output_scale_noise: float = 0.0,
        input_scale_noise: float = 0.0,
        uncertainty_bias: float = 0.5,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        shuffle_per_head: bool = True,
        prebuilt_anchor_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        recompute_in_backward: bool = False,
        exclusion_sets: Optional[list] = None,
        use_fp16_weights: bool = False,
        table_gating: bool = False,
        table_gating_init: float = -5.0,
        table_gating_leak: float = 0.01,
        return_per_table_outputs: bool = False,
    ):
        super().__init__()

        # Auto-compute tables_per_head for policies with fixed table counts.
        if anchor_sampling_policy == AnchorSamplingPolicy.HIERARCHICAL:
            tables_per_head = compute_hierarchical_n_tables(input_dim, n_anchor_pairs, tables_per_head)
        elif anchor_sampling_policy == AnchorSamplingPolicy.MULTISCALE:
            tables_per_head = compute_multiscale_n_tables(input_dim, n_anchor_pairs, tables_per_head)
        elif anchor_sampling_policy == AnchorSamplingPolicy.CONV2D:
            tables_per_head = compute_conv2d_n_tables(input_dim, tables_per_head)

        self.return_per_table_outputs = return_per_table_outputs
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
        self.normalize_weights = normalize_weights
        self.calibrate_output = calibrate_output
        self.output_scale_noise = output_scale_noise
        self.input_scale_noise = input_scale_noise
        self.uncertainty_bias = uncertainty_bias
        self.recompute_in_backward = recompute_in_backward or _FORCE_RECOMPUTE_IN_BACKWARD

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
            uncertainty_bias=uncertainty_bias,
            anchor_sampling_policy=anchor_sampling_policy,
            n_heads=n_heads,
            shuffle_per_head=shuffle_per_head,
            prebuilt_anchor_pairs=prebuilt_anchor_pairs,
            exclusion_sets=exclusion_sets,
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
            normalize_weights=normalize_weights,
            uncertainty_bias=uncertainty_bias,
            use_fp16_weights=use_fp16_weights,
        )

        # Table gating: learnable per-table gate with hard forward / soft backward (STE)
        self.table_gating = table_gating
        self.table_gating_leak = table_gating_leak
        if table_gating:
            self.gate_logits = nn.Parameter(
                torch.full((n_lookup_tables,), table_gating_init, device=device)
            )
            self._gate_temperature = 1.0
            self._gate_noise_scale = initial_weights_noise * 0.01


        if initial_weights_noise != 0.0:
            dev = device or torch.device("cpu")
            with torch.no_grad():
                rng_kwargs: dict = {"device": dev}
                if random_seed is not None:
                    rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed)
                self.projection.weights.add_(
                    torch.randn(self.projection.weights.shape, **rng_kwargs) * initial_weights_noise
                )
        self._internal_loss: Optional[torch.Tensor] = None

    def internal_loss(self) -> Optional[torch.Tensor]:
        """
        Returns the uncertainty regularization loss from the last forward pass (training only).

        The loss is mean(0.5 / (1 + |lookup_alt_deltas|)) over all alternative deltas —
        high when deltas are small (uncertain routing), low when deltas are large (confident).
        Penalizing this drives the network toward confident, hard-mode-like routing.

        Returns None if not in training mode or smooth_mode is False.
        """
        return self._internal_loss

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

        if self.training and self.input_scale_noise > 0.0:
            scale = 1.0 + (torch.rand(x.shape[0], 1, device=x.device) * 2 - 1) * self.input_scale_noise
            x = x * scale

        if self.training and self.recompute_in_backward and self.n_buckets == 1:
            # --- Fused path: AnchorPairsLookup + LProjection in one Function ---
            # Saves ~264 MB per block by recomputing lookup_indices in backward.
            if self.projection.normalize_weights:
                self.projection.apply_weight_column_norm()

            batch_size = x.shape[0]
            input_dim = x.shape[1]
            n_lookup_tables = self.n_heads * self.tables_per_head
            expected_len = batch_size * n_lookup_tables * self.n_alternatives
            if (
                self.lookup._cached_batch_offset is None
                or self.lookup._cached_batch_offset.numel() != expected_len
                or self.lookup._cached_batch_offset.device != x.device
            ):
                self.lookup._cached_batch_offset = (
                    torch.arange(batch_size, device=x.device, dtype=torch.long)
                    .repeat_interleave(n_lookup_tables * self.n_alternatives) * input_dim
                ).contiguous()

            output, lookup_alt_deltas = MultiHeadLutFunction.apply(
                x, self.projection.weights,
                self.lookup.anchor_pairs_a, self.lookup.anchor_pairs_b, self.lookup.powers,
                self.lookup.cmp_eps, self.n_alternatives, self.smooth_mode,
                0 if self.uncertainty_mode == UncertaintyMode.INVERSE_L1 else 1,
                self.uncertainty_bias,
                self.lookup._cached_batch_offset,
            )

            if self.uncertainty_mode == UncertaintyMode.INVERSE_L1:
                self._internal_loss = (self.uncertainty_bias / (1.0 + lookup_alt_deltas.abs())).mean()
            else:
                self._internal_loss = (self.uncertainty_bias / (1.0 + lookup_alt_deltas ** 2)).mean()

        else:
            # --- Original path ---
            return_alternatives = self.training or self.smooth_mode

            if self.training:
                (
                    lookup_indices, lookup_alt_indices, lookup_alt_deltas,
                    lookup_indices_grad_c, lookup_alt_indices_grad_c
                ) = self.lookup(x, return_alternatives=return_alternatives)
            else:
                lookup_indices, lookup_alt_indices, lookup_alt_deltas = self.lookup(
                    x, return_alternatives=return_alternatives)
                lookup_indices_grad_c = None
                lookup_alt_indices_grad_c = None

            if self.training and lookup_alt_deltas is not None:
                if self.uncertainty_mode == UncertaintyMode.INVERSE_L1:
                    self._internal_loss = (self.uncertainty_bias / (1.0 + lookup_alt_deltas.abs())).mean()
                else:
                    self._internal_loss = (self.uncertainty_bias / (1.0 + lookup_alt_deltas ** 2)).mean()
            else:
                self._internal_loss = None

            if self.n_buckets > 1:
                bucket_indices = bucket_indices.to(lookup_indices.dtype)
                lookup_indices = lookup_indices * self.n_buckets + bucket_indices.unsqueeze(1)
                if lookup_alt_indices is not None:
                    lookup_alt_indices = (
                        lookup_alt_indices * self.n_buckets + bucket_indices.unsqueeze(1).unsqueeze(2)
                    )

            output = self.projection(
                lookup_indices=lookup_indices,
                lookup_alt_indices=lookup_alt_indices,
                lookup_alt_deltas=lookup_alt_deltas,
                lookup_indices_grad_c=lookup_indices_grad_c,
                lookup_alt_indices_grad_c=lookup_alt_indices_grad_c,
            )

        B = x.shape[0]
        # Reshape: [B, n_lookup_tables, n_outputs] -> [B, n_heads, tables_per_head, n_outputs]
        output = output.view(B, self.n_heads, self.tables_per_head, self.n_outputs)
        # Apply table dropout: randomly zero entire tables during training
        if self.training and self.table_dropout > 0.0:
            mask = torch.bernoulli(
                torch.full((B, self.n_heads, self.tables_per_head, 1),
                           1.0 - self.table_dropout, device=output.device)
            )
            output = output * mask / (1.0 - self.table_dropout)
        if self.dropout is not None:
            output = self.dropout(output)

        if self.return_per_table_outputs:
            # Return raw per-table outputs without reduction.
            # Shape: [B, n_heads, tables_per_head, n_outputs]
            return output

        # Default: sum over tables_per_head -> [B, n_heads, n_outputs]
        output = output.sum(dim=2)
        if self.training and self.output_scale_noise > 0.0:
            # Multiply each (batch, head) by an independent U(1-s, 1+s) scalar
            scale = 1.0 + (torch.rand(output.shape[0], output.shape[1], 1, device=output.device) * 2 - 1) * self.output_scale_noise
            output = output * scale
        if self.calibrate_output:
            output = _calibrate_per_head_output(output)
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
      - Optional ``calibrate_output``: forwarded to ``MultiHeadLut`` (per-batch, per-head scaling over features at end of its forward).
    """

    def __init__(
        self,
        unfold_config: UnfoldConfiguration,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        fold_config: Optional[UnfoldConfiguration] = None,
        device: Optional[torch.device] = None,
        calibrate_output: bool = False,
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
            calibrate_output=calibrate_output,
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


class Conv2DLUT(nn.Module):
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
                f"Input channels C={C} do not match Conv2DLUT configuration "
                f"in_channels={self.in_channels}"
            )
        if (H, W) != (self.unfold_config.H, self.unfold_config.W):
            raise ValueError(
                f"Input spatial size {(H, W)} does not match Conv2DLUT configuration "
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


# Backward-compatible alias.
Conv2DLut = Conv2DLUT
