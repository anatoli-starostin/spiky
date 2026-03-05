"""
LUT-based cross-attention module.
"""
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from typing import Optional, Union

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.util.chunk_of_connections import ChunkOfConnections
from spiky.lutorch.lut_helpers import logarithmic_pe_buckets, rpe_matrix


class PairProcessingMode(str, Enum):
    LINEAR_COMBINATION = "linear_combination"
    CONCATENATION = "concatenation"


@dataclass
class PairProcessingConfig:
    """
    Configuration for (i, j) pair processing in LUTCrossAttention.
    
    mode:
        - LINEAR_COMBINATION: uses c1 * input1[i] + c2 * input2[j]
        - CONCATENATION: concatenates [input1[i], input2[j]]
    """
    mode: PairProcessingMode = PairProcessingMode.LINEAR_COMBINATION
    c1: float = 1.0
    c2: float = -2.0


class LUTCrossAttention(nn.Module):
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
        pair_config: Optional[PairProcessingConfig] = None,
    ):
        super().__init__()
        
        # Assert that MultiHeadLut has n_outputs=1
        assert multi_head_lut.n_outputs == 1, \
            f"LUTCrossAttention requires MultiHeadLut with n_outputs=1, got {multi_head_lut.n_outputs}"
        
        # Assert that MultiHeadLut has n_buckets matching n_positional_buckets
        assert multi_head_lut.n_buckets == n_positional_buckets, \
            f"MultiHeadLut.n_buckets ({multi_head_lut.n_buckets}) must match n_positional_buckets ({n_positional_buckets})"
        
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
        self.n_positional_buckets = n_positional_buckets
        
        # Cache for causal mask
        self._cached_causal_mask = None
        
        # Cache for RPE bucket indices [B, S, S]
        self._cached_bucket_indices = None
    
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
        
        # Create pair representation for all (i, j): [B, S, S, *]
        # input1[b, i, :] -> [B, S, 1, n_inputs] -> [B, S, S, n_inputs]
        input1_expanded = input1.unsqueeze(2).expand(batch_size, seq_len, seq_len, -1)  # [B, S, S, n_inputs]
        # input2[b, j, :] -> [B, 1, S, n_inputs] -> [B, S, S, n_inputs]
        input2_expanded = input2.unsqueeze(1).expand(batch_size, seq_len, seq_len, -1)  # [B, S, S, n_inputs]

        if self.pair_config.mode == PairProcessingMode.LINEAR_COMBINATION:
            # Linear combination: c1 * input1 + c2 * input2
            combined = (
                self.pair_config.c1 * input1_expanded
                + self.pair_config.c2 * input2_expanded
            )  # [B, S, S, n_inputs]
        else:
            # Concatenation: [input1, input2] along feature dimension
            combined = torch.cat([input1_expanded, input2_expanded], dim=-1)  # [B, S, S, 2 * n_inputs]
        
        # Reshape to [B * S * S, input_dim_for_lut]
        combined_flat = combined.view(-1, self.multi_head_lut.input_dim)  # [B * S * S, input_dim]
        
        # Compute bucket indices for relative positional encoding if needed
        bucket_indices = None
        if self.n_positional_buckets > 1:
            device = combined_flat.device
            cached = self._cached_bucket_indices
            if (
                cached is None
                or cached.shape[0] != batch_size
                or cached.shape[1] != seq_len
                or cached.shape[2] != seq_len
                or cached.device != device
            ):
                pe_buckets = logarithmic_pe_buckets(self.n_positional_buckets, seq_len, device)
                rpe = rpe_matrix(pe_buckets, seq_len, device).T  # [S, S]
                self._cached_bucket_indices = rpe.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, S, S]
            bucket_indices = self._cached_bucket_indices.view(-1)  # [B * S * S]
        
        # Apply MultiHeadLut: [B * S * S, n_inputs] -> [B * S * S, H, 1]
        lut_output = self.multi_head_lut(combined_flat, bucket_indices=bucket_indices)  # [B * S * S, H, 1]
        
        # Reshape to [B, S, S, H]
        attention_scores = lut_output.view(batch_size, seq_len, seq_len, self.n_heads, 1).squeeze(-1)  # [B, S, S, H]
        
        # Apply causal mask if needed
        if self.causal:
            device = attention_scores.device
            if (
                self._cached_causal_mask is None
                or self._cached_causal_mask.shape[1] != seq_len
                or self._cached_causal_mask.device != device
            ):
                # Create causal mask: mask[q, k] = 0 if k > q, else 1; broadcast with [B, S, S, H]
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(-1)  # [1, S, S, 1]
                self._cached_causal_mask = causal_mask
            else:
                causal_mask = self._cached_causal_mask
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax over key dimension (dim=2)
        attention_scores = attention_scores - attention_scores.max(dim=2, keepdim=True).values
        attention_scores = torch.softmax(attention_scores, dim=2)  # [B, S, S, H]
        
        return attention_scores
