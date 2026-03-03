"""
LUT-based cross-attention module.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Union

from multi_head_lut import MultiHeadLut
from spiky.util.chunk_of_connections import ChunkOfConnections


class LUTCrossAttention(nn.Module):
    """
    Cross-attention module using lookup tables.
    
    Takes two input sequences and computes attention scores using MultiHeadLut.
    
        Args:
        multi_head_lut: MultiHeadLut instance (must have n_outputs=1)
        c1: Coefficient for input1 in linear combination (default: 1.0)
        c2: Coefficient for input2 in linear combination (default: -2.0)
        causal: If True, apply causal mask to attention scores (default: True)
        n_positional_buckets: Number of positional buckets for relative positional encoding (default: 1).
                            If > 1, enables relative positional encoding via bucket indices.
    """
    
    def __init__(
        self,
        multi_head_lut: MultiHeadLut,
        c1: float = 1.0,
        c2: float = -2.0,
        causal: bool = True,
        n_positional_buckets: int = 1
    ):
        super().__init__()
        
        # Assert that MultiHeadLut has n_outputs=1
        assert multi_head_lut.n_outputs == 1, \
            f"LUTCrossAttention requires MultiHeadLut with n_outputs=1, got {multi_head_lut.n_outputs}"
        
        # Assert that MultiHeadLut has n_buckets matching n_positional_buckets
        assert multi_head_lut.n_buckets == n_positional_buckets, \
            f"MultiHeadLut.n_buckets ({multi_head_lut.n_buckets}) must match n_positional_buckets ({n_positional_buckets})"

        self.multi_head_lut = multi_head_lut
        self.n_inputs = multi_head_lut.input_dim
        self.n_heads = multi_head_lut.n_heads
        self.c1 = c1
        self.c2 = c2
        self.causal = causal
        self.n_positional_buckets = n_positional_buckets
        
        # Cache for causal mask
        self._cached_causal_mask = None
        
        # Cache for RPE matrix
        self._cached_rpe_matrix = None

    @staticmethod
    def _allocate_pe_buckets(num_buckets: int, seq_len: int) -> torch.Tensor:
        """
        Allocate positional encoding buckets.
        
        Similar to allocate_PE_buckets in spike_QK.ipynb:
        - For positions < B_half: bucket = position
        - For positions >= B_half: bucket = B_half + int(scale * log(pos / B_half))
        
        Args:
            num_buckets: Number of buckets
            seq_len: Sequence length
            
        Returns:
            Tensor of shape [seq_len] with bucket indices
        """
        pe_buckets = torch.zeros(seq_len, dtype=torch.long)
        if num_buckets <= 1:
            return pe_buckets
        
        B_half = num_buckets // 2
        for pos in range(seq_len):
            if pos < B_half:
                pe_buckets[pos] = pos
            else:
                log_term = math.log((pos + 1) / B_half)
                log_max_dist = math.log(seq_len / B_half)
                scale = (num_buckets - B_half) / log_max_dist
                log_bucket = B_half + int(scale * log_term)
                pe_buckets[pos] = min(log_bucket, num_buckets - 1)
        
        return pe_buckets
    
    @staticmethod
    def _allocate_rpe_matrix(buckets: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Allocate relative positional encoding matrix.
        
        Similar to allocate_RPE_matrix in spike_QK.ipynb:
        RPE[i, j] = buckets[max(0, i - j)]
        
        Args:
            buckets: Positional buckets tensor of shape [seq_len]
            seq_len: Current sequence length
            device: Device to place tensors on
            
        Returns:
            RPE matrix of shape [seq_len, seq_len] where RPE[i, j] = buckets[max(0, i - j)]
        """
        indices = torch.arange(seq_len, device=device)
        diff = indices.unsqueeze(1) - indices.unsqueeze(0)  # [seq_len, seq_len]
        diff = diff.clamp(min=0)  # Only non-negative differences
        diff = diff.clamp(max=len(buckets) - 1)  # Clamp to valid bucket range
        return buckets[diff]
    
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
            Attention scores tensor of shape [B, H, S, S]
        """
        batch_size, seq_len, _ = input1.shape
        assert input2.shape == input1.shape, f"input1 and input2 must have the same shape, got {input1.shape} and {input2.shape}"
        
        # Create linear combination: [B, S, S, n_inputs]
        # input1[b, i, :] -> [B, S, 1, n_inputs] -> [B, S, S, n_inputs]
        input1_expanded = input1.unsqueeze(2).expand(batch_size, seq_len, seq_len, -1)  # [B, S, S, n_inputs]
        # input2[b, j, :] -> [B, 1, S, n_inputs] -> [B, S, S, n_inputs]
        input2_expanded = input2.unsqueeze(1).expand(batch_size, seq_len, seq_len, -1)  # [B, S, S, n_inputs]
        
        # Linear combination: c1 * input1 + c2 * input2
        combined = self.c1 * input1_expanded + self.c2 * input2_expanded  # [B, S, S, n_inputs]
        
        # Reshape to [B * S * S, n_inputs]
        combined_flat = combined.view(-1, self.n_inputs)  # [B * S * S, n_inputs]
        
        # Compute bucket indices for relative positional encoding if needed
        bucket_indices = None
        if self.n_positional_buckets > 1:
            device = combined_flat.device
            # Get or create RPE matrix (cached for efficiency)
            if (
                self._cached_rpe_matrix is None
                or self._cached_rpe_matrix.shape[0] != seq_len
                or self._cached_rpe_matrix.device != device
            ):
                # Allocate PE buckets for current sequence length
                pe_buckets = self._allocate_pe_buckets(self.n_positional_buckets, seq_len)
                pe_buckets = pe_buckets.to(device)
                
                # Compute RPE matrix: [seq_len, seq_len]
                rpe_matrix = self._allocate_rpe_matrix(pe_buckets, seq_len, device)  # [S, S]
                
                self._cached_rpe_matrix = rpe_matrix
            else:
                rpe_matrix = self._cached_rpe_matrix
            
            # Expand to batch dimension: [B, S, S]
            rpe_expanded = rpe_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # [B, S, S]
            
            # Flatten to match combined_flat: [B * S * S]
            bucket_indices = rpe_expanded.view(-1)  # [B * S * S]
        
        # Apply MultiHeadLut: [B * S * S, n_inputs] -> [B * S * S, H, 1]
        lut_output = self.multi_head_lut(combined_flat, bucket_indices=bucket_indices)  # [B * S * S, H, 1]
        
        # Reshape to [B, H, S, S]
        attention_scores = lut_output.view(batch_size, seq_len, seq_len, self.n_heads, 1)  # [B, S, S, H, 1]
        attention_scores = attention_scores.squeeze(-1).permute(0, 3, 1, 2)  # [B, H, S, S]
        
        # Apply causal mask if needed
        if self.causal:
            # Get or create causal mask (cached for efficiency)
            device = attention_scores.device
            if (
                self._cached_causal_mask is None
                or self._cached_causal_mask.shape[-1] != seq_len
                or self._cached_causal_mask.device != device
            ):
                # Create causal mask: mask[i, j] = 0 if j > i, else 1
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # [S, S]
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
                self._cached_causal_mask = causal_mask
            else:
                causal_mask = self._cached_causal_mask
            
            # Set masked positions to -inf before softmax
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply softmax along the last dimension (by rows)
        attention_scores = torch.softmax(attention_scores, dim=-1)  # [B, H, S, S]
        
        return attention_scores
