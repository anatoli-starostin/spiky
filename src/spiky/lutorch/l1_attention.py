"""
L1-norm cross-attention module.
"""
import torch
import torch.nn as nn


class L1Attention(nn.Module):
    """
    Cross-attention module that uses L1 norm of input differences as attention scores.

    Unlike LUTCrossAttention, this module expects **multihead inputs**: each head has its
    own query/key vectors. For each head h and pair (i, j), score[i,j,h] = -||input1[i,h] - input2[j,h]||_1,
    then softmax over keys. No LUT, no pair-processing config, no positional buckets.

    Args:
        causal: If True, apply causal mask so that position k can only attend to positions j <= k.
        attention_temperature: Scale raw scores by 1/temperature before softmax (default: 1.0).
    """

    def __init__(
        self,
        causal: bool = True,
        attention_temperature: float = 1.0,
    ):
        super().__init__()
        self.causal = causal
        self.attention_temperature = float(attention_temperature)

        # Caches for causal path keyed by (batch_size, seq_len, device)
        self._cached_pair_meta = None
        self._cached_batched_rows = None
        self._cached_batched_cols = None
        self._cached_key_indices = None

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input1: Multihead input tensor of shape [B, S, H, D] (query per head).
            input2: Multihead input tensor of shape [B, S, H, D] (key per head).

        Returns:
            Attention scores tensor of shape [B, S, S, H].
        """
        batch_size, seq_len, H, head_dim = input1.shape
        assert input2.shape == input1.shape, (
            f"input1 and input2 must have the same shape, got {input1.shape} and {input2.shape}"
        )
        device = input1.device

        # [B*S, H, D]
        input1_flat = input1.view(batch_size * seq_len, H, head_dim)
        input2_flat = input2.view(batch_size * seq_len, H, head_dim)

        if self.causal:
            meta = (batch_size, seq_len, device)
            if self._cached_pair_meta != meta:
                rows_local, cols_local = torch.tril_indices(
                    seq_len, seq_len, offset=0, device=device
                )
                offsets = torch.arange(batch_size, device=device) * seq_len
                self._cached_batched_rows = (
                    rows_local.unsqueeze(0) + offsets.unsqueeze(1)
                ).reshape(-1)
                self._cached_batched_cols = (
                    cols_local.unsqueeze(0) + offsets.unsqueeze(1)
                ).reshape(-1)
                self._cached_key_indices = (self._cached_batched_cols % seq_len).contiguous()
                self._cached_pair_meta = meta

            batched_rows = self._cached_batched_rows
            batched_cols = self._cached_batched_cols

            q_vecs = input1_flat[batched_rows]   # [P, H, D]
            k_vecs = input2_flat[batched_cols]   # [P, H, D]
            diff = q_vecs - k_vecs               # [P, H, D]
            l1_per_pair_per_head = diff.abs().sum(dim=-1)  # [P, H]
            raw_scores = -l1_per_pair_per_head

            if self.attention_temperature != 1.0:
                raw_scores = raw_scores / self.attention_temperature

            dense_scores = torch.full(
                (batch_size * seq_len, seq_len, H),
                float("-inf"),
                device=device,
            )
            key_indices = self._cached_key_indices
            dense_scores[batched_rows, key_indices, :] = raw_scores

            attention_scores = dense_scores.view(batch_size, seq_len, seq_len, H)
            attention_scores = torch.softmax(attention_scores, dim=2)
        else:
            # [B, S, S, H, D]: input1[i,h] - input2[j,h] for all (i, j, h)
            input1_expanded = input1.unsqueeze(2).expand(batch_size, seq_len, seq_len, H, head_dim)
            input2_expanded = input2.unsqueeze(1).expand(batch_size, seq_len, seq_len, H, head_dim)
            diff = input1_expanded - input2_expanded  # [B, S, S, H, D]
            l1_grid = diff.abs().sum(dim=-1)  # [B, S, S, H]
            raw_scores = -l1_grid

            if self.attention_temperature != 1.0:
                raw_scores = raw_scores / self.attention_temperature

            attention_scores = torch.softmax(raw_scores, dim=2)

        return attention_scores
