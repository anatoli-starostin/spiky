"""
LUT-based cross-attention module.
"""
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from typing import Optional

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.util.chunk_of_connections import ChunkOfConnections
from spiky.lutorch.lut_helpers import logarithmic_pe_buckets, rpe_matrix, SelfExcitementMode
from spiky.lutorch.l_projection import _get_native_lutorch_manager, _USE_LUTORCH_CUSTOM_CUDA_KERNELS


class PairProcessingMode(str, Enum):
    LINEAR_COMBINATION = "linear_combination"
    CONCATENATION = "concatenation"


@dataclass(frozen=True)
class PairProcessingConfig:
    """
    Configuration for (i, j) pair processing in LUTAttention.
    
    mode:
        - LINEAR_COMBINATION: uses c1 * input1[i] + c2 * input2[j]
        - CONCATENATION: concatenates [input1[i], input2[j]]
    """
    mode: PairProcessingMode = PairProcessingMode.LINEAR_COMBINATION
    c1: float = 1.0
    c2: float = -2.0


class LUTAttention(nn.Module):
    """
    Cross-attention module using lookup tables.
    
    Takes two input sequences and computes attention scores using MultiHeadLut.
    
        Args:
        multi_head_lut: MultiHeadLut instance (must have n_outputs=1)
        causal: If True, apply causal mask to attention scores (default: True)
        n_positional_buckets: Number of positional buckets for relative positional encoding (default: 1).
                              If > 1, enables relative positional encoding via bucket indices.
        pair_config: Configuration of how (i, j) pairs are mapped to LUT inputs.
                     By default uses linear combination with c1=1.0, c2=-2.0.
    """
    
    def __init__(
        self,
        multi_head_lut: MultiHeadLut,
        causal: bool = True,
        n_positional_buckets: int = 1,
        include_diagonal: bool = True,
        pair_config: Optional[PairProcessingConfig] = None,
        do_sanity_checks: bool = False,
        attention_temperature: float = 1.0,
    ):
        super().__init__()
        
        # Assert that MultiHeadLut has n_outputs=1
        assert multi_head_lut.n_outputs == 1, \
            f"LUTAttention requires MultiHeadLut with n_outputs=1, got {multi_head_lut.n_outputs}"
        
        # Assert that MultiHeadLut has n_buckets matching n_positional_buckets
        assert multi_head_lut.n_buckets == n_positional_buckets, \
            f"MultiHeadLut.n_buckets ({multi_head_lut.n_buckets}) must match n_positional_buckets ({n_positional_buckets})"

        # Non-causal path: fall back to dense all-pairs computation.
        # For now, positional buckets are only supported in the causal path.
        if not causal and n_positional_buckets > 1:
            raise ValueError(
                "LUTAttention: n_positional_buckets > 1 is only supported when causal=True"
            )

        # Initialize pair-processing configuration (default: linear combination with c1=1.0, c2=-2.0)
        if pair_config is None:
            pair_config = PairProcessingConfig()
        self.pair_config = pair_config

        self.multi_head_lut = multi_head_lut
        self.n_heads = multi_head_lut.n_heads

        # In linear-combination mode, MultiHeadLut.input_dim == per-input feature dim.
        # In concatenation mode, MultiHeadLut.input_dim == 2 * per-input feature dim.
        if self.pair_config.mode == PairProcessingMode.CONCATENATION:
            assert multi_head_lut.input_dim % 2 == 0, \
                f"MultiHeadLut.input_dim ({multi_head_lut.input_dim}) must be even in CONCATENATION mode"
            self.n_inputs = multi_head_lut.input_dim // 2
        else:
            self.n_inputs = multi_head_lut.input_dim

        self.causal = causal
        self.include_diagonal = bool(include_diagonal)
        self.n_positional_buckets = n_positional_buckets
        self.do_sanity_checks = do_sanity_checks
        self.attention_temperature = float(attention_temperature)

        # Caches for causal sparse pairs (batched rows, cols, key indices, and rpe) keyed by (batch_size, seq_len, device)
        self._cached_pair_meta = None  # (batch_size, seq_len, device)
        self._cached_batched_rows = None
        self._cached_batched_cols = None
        self._cached_key_indices = None
        self._cached_bucket_indices = None
        # Cache for dense bucket indices in non-causal path keyed by (batch_size, seq_len, device)
        self._cached_dense_bucket_meta = None
        self._cached_dense_bucket_indices = None
    
    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input1: Input tensor of shape [B, S, n_inputs]
            input2: Input tensor of shape [B, S, n_inputs]
            
        Returns:
            Attention scores tensor of shape [B, S, S, H]
        """
        batch_size, seq_len, feature_dim = input1.shape
        assert input2.shape == input1.shape, f"input1 and input2 must have the same shape, got {input1.shape} and {input2.shape}"
        assert feature_dim == self.n_inputs, \
            f"Expected input feature dim {self.n_inputs}, got {feature_dim}"

        device = input1.device
        H = self.n_heads

        # Flatten inputs: [B * S, n_inputs]
        input1_flat = input1.view(batch_size * seq_len, feature_dim)
        input2_flat = input2.view(batch_size * seq_len, feature_dim)

        if self.causal:
            # Build or reuse lower-triangular indices and RPE pairs for this (B, S, device, include_diagonal)
            # - include_diagonal=True  -> allow self: k <= q  (offset=0)
            # - include_diagonal=False -> strictly causal: k < q (offset=-1)
            meta = (batch_size, seq_len, device, self.include_diagonal)
            if self._cached_pair_meta != meta:
                offset = 0 if self.include_diagonal else -1
                rows_local, cols_local = torch.tril_indices(
                    seq_len, seq_len, offset=offset, device=device
                )  # [num_pairs_single] with k <= q (or k < q if exclude diagonal)

                offsets = torch.arange(batch_size, device=device) * seq_len  # [B]
                self._cached_batched_rows = (
                    rows_local.unsqueeze(0) + offsets.unsqueeze(1)
                ).reshape(-1)  # [P], where P = B * num_pairs_single
                self._cached_batched_cols = (
                    cols_local.unsqueeze(0) + offsets.unsqueeze(1)
                ).reshape(-1)  # [P]

                # Within-sequence key indices for scattering into [B*S, S, H]
                self._cached_key_indices = (self._cached_batched_cols % seq_len).contiguous()  # [P]

                if self.n_positional_buckets > 1:
                    pe_buckets = logarithmic_pe_buckets(self.n_positional_buckets, seq_len, device)
                    # When include_diagonal=False we never have distance 0. Shift buckets so
                    # distance 1 -> bucket 0, distance 2 -> bucket 1, ... to use the full weight budget.
                    if not self.include_diagonal and seq_len > 1:
                        buckets = torch.cat([pe_buckets[0:1], pe_buckets[0:-1]], dim=0)
                    else:
                        buckets = pe_buckets
                    rpe = rpe_matrix(buckets, seq_len, device)  # [S, S]
                    rpe_pairs = rpe[rows_local, cols_local]  # [num_pairs_single]
                    if self.do_sanity_checks:
                        # Sanity check: rpe_pairs must agree with distance-based buckets
                        dist = (rows_local - cols_local).clamp(min=0, max=buckets.shape[0] - 1)
                        expected_pairs = buckets[dist]
                        assert torch.equal(rpe_pairs, expected_pairs), "rpe_pairs inconsistent with pe_buckets and (q,k) distance"
                    self._cached_bucket_indices = rpe_pairs.repeat(batch_size).contiguous()  # [P]
                    if self.do_sanity_checks:
                        # Sanity check: repeated buckets are identical across batches
                        P_single = rpe_pairs.shape[0]
                        P = self._cached_bucket_indices.shape[0]
                        assert P % P_single == 0, "bucket_indices length mismatch with per-sequence pairs"
                        cached_view = self._cached_bucket_indices.view(batch_size, P_single)
                        assert torch.all(cached_view[0] == cached_view), "bucket_indices differ between batches"
                else:
                    self._cached_bucket_indices = None

                self._cached_pair_meta = meta

            batched_rows = self._cached_batched_rows
            batched_cols = self._cached_batched_cols

            # Build pair representations only for valid (q, k); P = total number of valid (q,k) pairs across batch
            q_vecs = input1_flat[batched_rows]  # [P, n_inputs]
            k_vecs = input2_flat[batched_cols]  # [P, n_inputs]

            if self.pair_config.mode == PairProcessingMode.LINEAR_COMBINATION:
                combined_flat = (
                    self.pair_config.c1 * q_vecs + self.pair_config.c2 * k_vecs
                )  # [P, n_inputs]
            else:
                combined_flat = torch.cat([q_vecs, k_vecs], dim=-1)  # [P, 2 * n_inputs]

            # Bucket indices for relative positional encoding if needed
            bucket_indices = self._cached_bucket_indices

            # Apply MultiHeadLut: [P, n_inputs] -> [P, H, 1]
            lut_output = self.multi_head_lut(
                combined_flat, bucket_indices=bucket_indices
            )  # [P, H, 1]
            raw_scores = lut_output.squeeze(-1)  # [P, H]
            if self.attention_temperature != 1.0:
                raw_scores = raw_scores / self.attention_temperature

            # Densify into [B * S, S, H] filled with -inf, scatter valid scores
            dense_scores = torch.full(
                (batch_size * seq_len, seq_len, H),
                float("-inf"),
                device=device,
            )
            key_indices = self._cached_key_indices  # [P]
            dense_scores[batched_rows, key_indices, :] = raw_scores  # [P, H]

            # Reshape to [B, S, S, H]
            attention_scores = dense_scores.view(batch_size, seq_len, seq_len, H)

            # Special-case q=0 when include_diagonal=False: there are no valid keys under the
            # strict causal mask (k < q). Let it attend deterministically to itself (non-inplace
            # so autograd is not broken).
            if not self.include_diagonal and seq_len > 0:
                q0_k0 = (
                    (torch.arange(seq_len, device=device) == 0).view(1, 1, seq_len, 1).expand(batch_size, seq_len, seq_len, H)
                    & (torch.arange(seq_len, device=device) == 0).view(1, seq_len, 1, 1).expand(batch_size, seq_len, seq_len, H)
                )
                attention_scores = attention_scores.masked_fill(q0_k0, 0.0)

            # Apply numerically stable softmax over keys
            attention_scores = torch.softmax(attention_scores, dim=2)  # [B, S, S, H]
        else:
            # Create pair representation for all (i, j): [B, S, S, *]
            input1_expanded = input1.unsqueeze(2).expand(batch_size, seq_len, seq_len, -1)  # [B, S, S, n_inputs]
            input2_expanded = input2.unsqueeze(1).expand(batch_size, seq_len, seq_len, -1)  # [B, S, S, n_inputs]

            if self.pair_config.mode == PairProcessingMode.LINEAR_COMBINATION:
                combined = (
                    self.pair_config.c1 * input1_expanded
                    + self.pair_config.c2 * input2_expanded
                )  # [B, S, S, n_inputs]
            else:
                combined = torch.cat([input1_expanded, input2_expanded], dim=-1)  # [B, S, S, 2 * n_inputs]

            combined_flat = combined.view(-1, self.multi_head_lut.input_dim)  # [B * S * S, input_dim]

            # No positional buckets in non-causal mode (enforced above), so bucket_indices stays None.
            lut_output = self.multi_head_lut(combined_flat, bucket_indices=None)  # [B * S * S, H, 1]
            attention_scores = lut_output.view(batch_size, seq_len, seq_len, H, 1).squeeze(-1)  # [B, S, S, H]
            if self.attention_temperature != 1.0:
                attention_scores = attention_scores / self.attention_temperature
            attention_scores = torch.softmax(attention_scores, dim=2)

        return attention_scores


class LUTAttnFwdBwdNa1(torch.autograd.Function):
    """
    Fused autograd function for LUTAttentionV2 forward+backward.
    Handles n_alternatives=1, non-smooth mode, single pass over pairs on the GPU.
    """

    @staticmethod
    def forward(ctx, x, weights, anchor_a, anchor_b, pair_rows, pair_cols, rel_pe,
                H, tables_per_head, causal, self_excitement, cmp_eps, uncertainty_bias,
                se_mode_int):
        mgr = _get_native_lutorch_manager()
        pair_out_buf, result = mgr.lut_attn_fwd_na1(
            x.contiguous(), weights, anchor_a, anchor_b,
            pair_rows, pair_cols, rel_pe,
            H, tables_per_head, causal, self_excitement, float(cmp_eps),
            se_mode_int,
        )
        ctx.save_for_backward(x, weights, anchor_a, anchor_b, pair_rows, pair_cols, rel_pe, pair_out_buf)
        ctx.H = H
        ctx.tables_per_head = tables_per_head
        ctx.causal = causal
        ctx.self_excitement = self_excitement
        ctx.cmp_eps = cmp_eps
        ctx.uncertainty_bias = uncertainty_bias
        ctx.se_mode_int = se_mode_int
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, weights, anchor_a, anchor_b, pair_rows, pair_cols, rel_pe, pair_out_buf = ctx.saved_tensors
        mgr = _get_native_lutorch_manager()
        x_grad, weights_grad, rel_pe_grad = mgr.lut_attn_bwd_na1(
            x, weights, anchor_a, anchor_b,
            pair_rows, pair_cols, rel_pe,
            pair_out_buf, grad_output.contiguous(),
            ctx.H, ctx.tables_per_head, ctx.causal, ctx.self_excitement,
            float(ctx.cmp_eps), float(ctx.uncertainty_bias),
            ctx.se_mode_int,
        )
        return (x_grad, weights_grad, None, None, None, None, rel_pe_grad,
                None, None, None, None, None, None, None)


class LUTAttentionV2(nn.Module):
    """
    Pairwise LUT attention with explicit relative positional encoding.

    For a sequence of T tokens, enumerates M valid position pairs (i, j),
    concatenates [x_i, x_j, RelPE[dist(i,j)]], and feeds through a MultiHeadLut.
    The results are summed over all M pairs to produce [B, H, O].

    Pair enumeration:
        causal=True,  include_diagonal=True  → j ≤ i   (M = T*(T+1)/2)
        causal=True,  include_diagonal=False → j < i   (M = T*(T-1)/2)
        causal=False, include_diagonal=True  → all     (M = T^2)
        causal=False, include_diagonal=False → i ≠ j   (M = T*(T-1))

    RelPE indexing:
        Causal: dist = i - j  ≥ 0;  rel_pe[0] = diagonal, rel_pe[k] = distance k, …
        Non-causal: dist = |i - j|; rel_pe[0] = diagonal, rel_pe[k] = distance k, …
        Required size: T (with diagonal) or T-1 (without diagonal).

    Args:
        multi_head_lut: Pre-built MultiHeadLut with input_dim = 2*E + pos_dim,
                        n_heads = H, n_outputs = O.
        seq_len:        Fixed sequence length used to pre-compute pair indices.
        causal:         If True, only include pairs with j ≤ i (or j < i).
        include_diagonal: If True, include pairs where i == j.
    """

    def __init__(
        self,
        multi_head_lut: MultiHeadLut,
        seq_len: int,
        causal: bool = True,
        include_diagonal: bool = True,
        self_excitement: bool = False,
        self_excitement_mode: SelfExcitementMode = SelfExcitementMode.LINEAR,
    ):
        super().__init__()
        self.multi_head_lut = multi_head_lut
        self.causal = causal
        self.include_diagonal = include_diagonal
        self.self_excitement = self_excitement
        self.self_excitement_mode = self_excitement_mode

        rows, cols = [], []
        for i in range(seq_len):
            for j in range(seq_len):
                if causal and j > i:
                    continue
                if not include_diagonal and i == j:
                    continue
                rows.append(i)
                cols.append(j)

        self.register_buffer('pair_rows', torch.tensor(rows, dtype=torch.long))
        self.register_buffer('pair_cols', torch.tensor(cols, dtype=torch.long))

    def forward(self, x: torch.Tensor, rel_pe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      Input sequence [B, T, E].
            rel_pe: Relative position embeddings [D, pos_dim] where D ≥ max distance + 1.
                    rel_pe[0] = same position (diagonal), rel_pe[k] = distance k.

        Returns:
            [B, T, H, O] — per-token scatter-sum of MultiHeadLut outputs.
                           Output at position i = sum of LUT outputs over all pairs (i, j).
        """
        B, T, E = x.shape
        M = self.pair_rows.shape[0]

        # --- Fast path: fused CUDA kernel (n_alternatives=1, non-smooth) ---
        lut = self.multi_head_lut
        H = lut.n_heads
        _mgr = _get_native_lutorch_manager() if _USE_LUTORCH_CUSTOM_CUDA_KERNELS else None
        if (
            _mgr is not None
            and x.is_cuda
            and x.dtype == torch.float32
            and not lut.smooth_mode
            and lut.n_alternatives == 1
            and lut.n_buckets == 1
            and lut.table_dropout == 0.0
            and lut.scatter_indices is None
            and lut.post_processor is None
            and not lut.calibrate_output
            and not lut.projection.normalize_weights
        ):
            if lut.training and lut.input_scale_noise > 0.0:
                scale = 1.0 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * lut.input_scale_noise
                x = x * scale
            se_mode_int = {
                SelfExcitementMode.LINEAR: 0,
                SelfExcitementMode.QUADRATIC: 1,
                SelfExcitementMode.EXPONENTIAL: 2,
            }[self.self_excitement_mode]
            result = LUTAttnFwdBwdNa1.apply(
                x.contiguous(), lut.projection.weights,
                lut.lookup.anchor_pairs_a, lut.lookup.anchor_pairs_b,
                self.pair_rows, self.pair_cols, rel_pe,
                H, lut.tables_per_head, self.causal, self.self_excitement,
                float(lut.lookup.cmp_eps), float(lut.lookup.uncertainty_bias),
                se_mode_int,
            )
            return result  # [B, T, H, O]
        # --- End fast path ---

        x_i = x[:, self.pair_rows, :]   # [B, M, E]
        x_j = x[:, self.pair_cols, :]   # [B, M, E]

        # Distance: non-negative for causal; |i-j| for non-causal
        if self.causal:
            dists = self.pair_rows - self.pair_cols   # [M], range [0, T-1]
        else:
            dists = (self.pair_rows - self.pair_cols).abs()  # [M]

        rpe = rel_pe[dists].unsqueeze(0).expand(B, -1, -1)  # [B, M, pos_dim]

        # Concatenate and flatten: [B*M, 2E + pos_dim]
        pairs = torch.cat([x_i, x_j, rpe], dim=-1)
        pairs_flat = pairs.reshape(B * M, -1)

        # Apply LUT: [B*M, H, O]
        out = self.multi_head_lut(pairs_flat)
        H, O = out.shape[1], out.shape[2]

        # Scatter-sum by query position: [B, M, H*O] → [B, T, H*O] → [B, T, H, O]
        if self.self_excitement:
            # Per-head SE: scale each head's outputs based on mean absolute activation.
            s = out.abs().mean(dim=-1, keepdim=True)  # [B*M, H, 1]
            if self.self_excitement_mode == SelfExcitementMode.LINEAR:
                out = out * s
            elif self.self_excitement_mode == SelfExcitementMode.QUADRATIC:
                out = out * s * s
            elif self.self_excitement_mode == SelfExcitementMode.EXPONENTIAL:
                out = out * torch.exp(s)
        out_flat = out.reshape(B, M, H * O)
        result = torch.zeros(B, T, H * O, device=x.device, dtype=x.dtype)
        idx = self.pair_rows.unsqueeze(0).unsqueeze(-1).expand(B, M, H * O)
        result.scatter_add_(1, idx, out_flat)
        return result.reshape(B, T, H, O)
