import torch
import torch.nn as nn


def explicit_mmul(lhs, rhs):
    """
    Explicit pairwise dot-product matrix multiplication for tensors of shape (B, H, S, D).

    Args:
        lhs: Tensor with shape (B, H, S, D).
        rhs: Tensor with shape (B, H, S, D).
    Returns:
        Tensor with shape (B, H, S, S), where output[..., i, j] = dot(lhs[..., i, :], rhs[..., j, :]).
    """
    if lhs.dim() != 4 or rhs.dim() != 4:
        raise ValueError("lhs and rhs must both have shape (B, H, S, D)")
    if lhs.shape != rhs.shape:
        raise ValueError("lhs and rhs must have identical shapes (B, H, S, D)")

    batch_size, num_heads, seq_len, head_dim = lhs.shape

    # Row-expanded lhs: (B, H, S, D) -> (B, H, S, S, D) -> (B, H, S * S, D).
    lhs_rows = lhs.unsqueeze(3).expand(batch_size, num_heads, seq_len, seq_len, head_dim)
    lhs_rows = lhs_rows.reshape(batch_size, num_heads, seq_len * seq_len, head_dim)

    # Column-expanded rhs: (B, H, S, D) -> (B, H, S, S, D) -> (B, H, S * S, D).
    rhs_cols = rhs.unsqueeze(2).expand(batch_size, num_heads, seq_len, seq_len, head_dim)
    rhs_cols = rhs_cols.reshape(batch_size, num_heads, seq_len * seq_len, head_dim)

    # Elementwise multiply + sum over D: (B, H, S * S, D) -> (B, H, S * S) -> (B, H, S, S).
    pairwise_dots = (lhs_rows * rhs_cols).sum(dim=-1)
    return pairwise_dots.view(batch_size, num_heads, seq_len, seq_len)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1, device=None):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.w_k = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.w_v = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.w_o = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        # x: (B, S, E), causal_mask: (1, 1, S, S)
        batch_size, seq_len, _ = x.shape

        # Projection matrix multiplications for Q, K, V: (B, S, E) -> (B, S, E).
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Split into heads: (B, S, E) -> (B, H, S, D), where D = E / H.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Explicit scores matrix multiplication: (B, H, S, D) x (B, H, S, D) -> (B, H, S, S).
        scores = explicit_mmul(q, k) * self.scale
        # Masked scores: (B, H, S, S).
        scores = scores.masked_fill(causal_mask, float("-inf"))
        # Attention weights: (B, H, S, S).
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output matrix multiplication: (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D).
        attended = attn_weights @ v
        # Merge heads: (B, H, S, D) -> (B, S, E).
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)

        # Output projection matrix multiplication: (B, S, E) -> (B, S, E).
        return self.w_o(attended)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim=128, dropout=0.1, device=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim, device=device)
        self.norm2 = nn.LayerNorm(embedding_dim, device=device)
        self.self_attn = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
        )
        self.ffn_in = nn.Linear(embedding_dim, ffn_dim, device=device)
        self.ffn_out = nn.Linear(ffn_dim, embedding_dim, device=device)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, causal_mask):
        # x: (B, S, E), causal_mask: (1, 1, S, S).
        # Pre-norm attention block with residual path.
        # attn_in: (B, S, E).
        attn_in = self.norm1(x)
        # attn_out: (B, S, E).
        attn_out = self.self_attn(attn_in, causal_mask=causal_mask)
        # Residual result: (B, S, E).
        x = x + self.dropout(attn_out)

        # Pre-norm FFN block with two explicit matrix multiplications.
        # ffn_in: (B, S, E).
        ffn_in = self.norm2(x)
        # ffn_hidden: (B, S, F), where F = ffn_dim.
        ffn_hidden = self.ffn_in(ffn_in)
        ffn_hidden = self.activation(ffn_hidden)
        ffn_hidden = self.dropout(ffn_hidden)
        # ffn_out: (B, S, E).
        ffn_out = self.ffn_out(ffn_hidden)
        # Residual result: (B, S, E).
        x = x + self.dropout(ffn_out)
        return x


class VanillaTransformer(nn.Module):
    """
    A simple decoder-only Transformer sketch with trainable positional embeddings.
    """

    def __init__(
        self,
        vocab_size,
        context_size,
        num_layers=6,
        num_heads=4,
        embedding_dim=32,
        ffn_dim=128,
        dropout=0.1,
        device=None,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        if device is None:
            device = torch.device("cpu")

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, context_size, embedding_dim, device=device)
        )

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim, device=device)
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False, device=device)

    def _causal_mask(self, seq_len, device):
        # Returns causal mask with shape (1, 1, S, S), broadcast over batch and heads.
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len)

    def forward(self, tokens):
        """
        Args:
            tokens: Tensor of shape (B, S), with S <= context_size.
        Returns:
            logits: Tensor of shape (B, S, vocab_size).
        """
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape (batch_size, seq_len)")

        _, seq_len = tokens.shape
        if seq_len > self.context_size:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds context_size {self.context_size}"
            )

        # Token embeddings: (B, S) -> (B, S, E).
        x = self.token_embedding(tokens)
        # Positional slice: (1, S, E); broadcast add gives (B, S, E).
        x = x + self.positional_embedding[:, :seq_len, :]
        # Dropout keeps shape: (B, S, E).
        x = self.dropout(x)

        # Causal mask: (1, 1, S, S).
        mask = self._causal_mask(seq_len=seq_len, device=tokens.device)
        # Each block keeps shape: (B, S, E) -> (B, S, E).
        for layer in self.layers:
            x = layer(x, causal_mask=mask)
        # Final norm keeps shape: (B, S, E).
        x = self.final_norm(x)
        # Output projection: (B, S, E) -> (B, S, vocab_size).
        logits = self.output_head(x)

        return logits

