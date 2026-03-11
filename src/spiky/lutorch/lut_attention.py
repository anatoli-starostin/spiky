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
from spiky.lutorch.lut_helpers import logarithmic_pe_buckets, rpe_matrix


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
