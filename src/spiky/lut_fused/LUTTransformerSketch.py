import torch
import torch.nn as nn

from spiky.lut_fused.LUTLayer import LUTLayer, ProjectionLUTLayer, SynapseMeta, GradientPolicy
from spiky.util.synapse_growth import PointSamplingPolicy, PointSamplingType


def explicit_lut_mmul(lhs, rhs):
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
    pairwise_dots = -(lhs_rows - rhs_cols).abs().sum(dim=-1)
    return pairwise_dots.view(batch_size, num_heads, seq_len, seq_len)


def explicit_lut_mmul_with_pair_lut(lhs, rhs, pair_lut):
    """
    Explicit pairwise score computation for tensors of shape (B, H, S, D) using a LUT scorer.

    Instead of distance reduction on (lhs_rows - rhs_cols), this function concatenates
    lhs_rows and rhs_cols on the last dimension and applies a LUTLayer with n_outputs=D.

    Args:
        lhs: Tensor with shape (B, H, S, D).
        rhs: Tensor with shape (B, H, S, D).
        pair_lut: LUTLayer that takes input size (2 * D) and returns D output values.
    Returns:
        Tensor with shape (B, H, S, S).
    """
    if lhs.dim() != 4 or rhs.dim() != 4:
        raise ValueError("lhs and rhs must both have shape (B, H, S, D)")
    if lhs.shape != rhs.shape:
        raise ValueError("lhs and rhs must have identical shapes (B, H, S, D)")
    _, _, _, head_dim = lhs.shape
    if not isinstance(pair_lut, LUTLayer):
        raise ValueError("pair_lut must be an instance of LUTLayer")
    if pair_lut.n_outputs() != head_dim:
        raise ValueError("pair_lut must be constructed with n_outputs=head_dim")

    batch_size, num_heads, seq_len, head_dim = lhs.shape

    # Row-expanded lhs: (B, H, S, D) -> (B, H, S, S, D) -> (B, H, S * S, D).
    lhs_rows = lhs.unsqueeze(3).expand(batch_size, num_heads, seq_len, seq_len, head_dim)
    lhs_rows = lhs_rows.reshape(batch_size, num_heads, seq_len * seq_len, head_dim)

    # Column-expanded rhs: (B, H, S, D) -> (B, H, S, S, D) -> (B, H, S * S, D).
    rhs_cols = rhs.unsqueeze(2).expand(batch_size, num_heads, seq_len, seq_len, head_dim)
    rhs_cols = rhs_cols.reshape(batch_size, num_heads, seq_len * seq_len, head_dim)

    # Concatenate pairs: (B, H, S * S, D) + (B, H, S * S, D) -> (B, H, S * S, 2D).
    pair_features = torch.cat([lhs_rows, rhs_cols], dim=-1)

    # LUT expects shape (N, 1, F). Here N = B * H * S * S, F = 2D.
    pair_features = pair_features.reshape(batch_size * num_heads * seq_len * seq_len, 1, 2 * head_dim)
    pair_scores = pair_lut(pair_features)

    # pair_scores: (B * H * S * S, 1, D) -> scalar score via sum over D -> (B, H, S, S).
    pair_scores = pair_scores.sum(dim=-1)
    return pair_scores.reshape(batch_size, num_heads, seq_len, seq_len)


class MultiHeadLUTSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1,
        lut_n_detectors=8,
        lut_n_anchors_per_detector=4,
        projection_sampling_policy: PointSamplingPolicy = None,
        use_pair_lut_mmul=False,
        normalize_scores_by_max=True,
        score_norm_eps=1e-6,
        gradient_policy: GradientPolicy = None,
        lut_synapse_meta=SynapseMeta(),
        device=None,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_pair_lut_mmul = use_pair_lut_mmul
        # For L1-sum scores (negative Manhattan distance), use 1 / D normalization.
        self.scale = self.head_dim ** -1.0
        self.normalize_scores_by_max = normalize_scores_by_max
        self.score_norm_eps = score_norm_eps

        # Place one detector-group center per head chunk along output width.
        if projection_sampling_policy is None:
            projection_sampling_policy = PointSamplingPolicy(
                PointSamplingType.Grid,
                pad_h=0.0,
                pad_w=(self.head_dim - 1) / 2.0,
                grid_h=1,
                grid_w=self.num_heads,
                stride_h=0,
                stride_w=self.head_dim,
            )

        # Q/K/V are projected with per-head detector groups over (1, embedding_dim) -> (1, embedding_dim).
        self.w_q = ProjectionLUTLayer(
            input_shape=(1, embedding_dim),
            output_shape=(1, embedding_dim),
            n_anchors_per_detector=lut_n_anchors_per_detector,
            n_detector_groups=num_heads,
            n_detectors_in_group=lut_n_detectors,
            receptive_shape=(1, embedding_dim * 10),
            projection_shape=(1, self.head_dim),
            detectors_sampling_policy=projection_sampling_policy,
            weights_gradient_policy=gradient_policy,
            synapse_meta=lut_synapse_meta,
            device=device,
        )
        self.w_k = ProjectionLUTLayer(
            input_shape=(1, embedding_dim),
            output_shape=(1, embedding_dim),
            n_anchors_per_detector=lut_n_anchors_per_detector,
            n_detector_groups=num_heads,
            n_detectors_in_group=lut_n_detectors,
            receptive_shape=(1, embedding_dim * 10),
            projection_shape=(1, self.head_dim),
            detectors_sampling_policy=projection_sampling_policy,
            weights_gradient_policy=gradient_policy,
            synapse_meta=lut_synapse_meta,
            device=device,
        )
        self.w_v = ProjectionLUTLayer(
            input_shape=(1, embedding_dim),
            output_shape=(1, embedding_dim),
            n_anchors_per_detector=lut_n_anchors_per_detector,
            n_detector_groups=num_heads,
            n_detectors_in_group=lut_n_detectors,
            receptive_shape=(1, embedding_dim * 10),
            projection_shape=(1, self.head_dim),
            detectors_sampling_policy=projection_sampling_policy,
            weights_gradient_policy=gradient_policy,
            synapse_meta=lut_synapse_meta,
            device=device,
        )
        self.w_o = LUTLayer(
            n_inputs=embedding_dim,
            n_outputs=embedding_dim,
            n_detectors=lut_n_detectors,
            n_anchors_per_detector=lut_n_anchors_per_detector,
            weights_gradient_policy=gradient_policy,
            synapse_meta=lut_synapse_meta,
            sequence_length=1,
            device=device,
        )
        self.pair_score_lut = None
        if self.use_pair_lut_mmul:
            self.pair_score_lut = LUTLayer(
                n_inputs=2 * self.head_dim,
                n_outputs=self.head_dim,
                n_detectors=lut_n_detectors,
                n_anchors_per_detector=lut_n_anchors_per_detector,
                weights_gradient_policy=gradient_policy,
                synapse_meta=lut_synapse_meta,
                sequence_length=1,
                device=device,
            )
        self.attn_dropout = nn.Dropout(dropout)

    def set_external_learning_rate_hook(self, lr_hook):
        self.w_q.set_external_learning_rate_hook(lr_hook)
        self.w_k.set_external_learning_rate_hook(lr_hook)
        self.w_v.set_external_learning_rate_hook(lr_hook)
        self.w_o.set_external_learning_rate_hook(lr_hook)
        if self.pair_score_lut is not None:
            self.pair_score_lut.set_external_learning_rate_hook(lr_hook)

    @staticmethod
    def _apply_lut_3d(lut_layer, x):
        # x: (B, S, N) -> (B * S, N) -> LUT expects (B * S, 1, N) -> (B, S, M).
        batch_size, seq_len, n_inputs = x.shape
        x_flat = x.reshape(batch_size * seq_len, 1, n_inputs)
        out_flat = lut_layer(x_flat).reshape(batch_size, seq_len, -1)
        return out_flat

    @staticmethod
    def _apply_projection_lut_3d(proj_lut_layer, x):
        # x: (B, S, E) -> (B * S, 1, 1, E) for ProjectionLUTLayer with input_shape=(1, E).
        batch_size, seq_len, embedding_dim = x.shape
        x_proj = x.reshape(batch_size * seq_len, 1, 1, embedding_dim)
        out_proj = proj_lut_layer(x_proj)
        # out_proj: (B * S, 1, 1, E) -> (B, S, E).
        return out_proj.reshape(batch_size, seq_len, embedding_dim)

    def forward(self, x, causal_mask):
        # x: (B, S, E), causal_mask: (1, 1, S, S)
        batch_size, seq_len, _ = x.shape

        # Projection matrix multiplications for Q, K, V: (B, S, E) -> (B, S, E).
        q = self._apply_projection_lut_3d(self.w_q, x)
        k = self._apply_projection_lut_3d(self.w_k, x)
        v = self._apply_projection_lut_3d(self.w_v, x)

        # Split into heads: (B, S, E) -> (B, H, S, D), where D = E / H.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Explicit scores matrix multiplication: (B, H, S, D) x (B, H, S, D) -> (B, H, S, S).
        if self.use_pair_lut_mmul:
            scores = explicit_lut_mmul_with_pair_lut(q, k, self.pair_score_lut) * self.scale
        else:
            scores = explicit_lut_mmul(q, k) * self.scale
        if self.normalize_scores_by_max:
            # Normalize each query row by its max absolute score magnitude.
            denom = scores.abs().amax(dim=-1, keepdim=True).clamp_min(self.score_norm_eps).detach()
            scores = scores / denom
        # Masked scores: (B, H, S, S).
        scores_max = scores.amax(dim=-1, keepdim=True)
        scores = scores - scores_max
        scores = scores.masked_fill(causal_mask, float("-inf"))
        # Attention weights: (B, H, S, S).
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output matrix multiplication: (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D).
        attended = attn_weights @ v
        # Merge heads: (B, H, S, D) -> (B, S, E).
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)

        # Output projection matrix multiplication: (B, S, E) -> (B, S, E).
        return self._apply_lut_3d(self.w_o, attended)


class LUTTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim=32,
        num_heads=4,
        ffn_dim=32,
        dropout=0.1,
        attn_lut_n_detectors=16,
        attn_lut_n_anchors_per_detector=10,
        projection_sampling_policy: PointSamplingPolicy = None,
        use_pair_lut_mmul=False,
        ffn_lut_n_detectors=16,
        ffn_lut_n_anchors_per_detector=10,
        normalize_scores_by_max=True,
        score_norm_eps=1e-6,
        rezero_init=0.1,
        gradient_policy: GradientPolicy = None,
        lut_synapse_meta=SynapseMeta(),
        device=None,
    ):
        super().__init__()
        # ReZero-style residual gates help stabilize training without LayerNorm.
        if rezero_init is not None:
            self.alpha_attn = nn.Parameter(torch.tensor(float(rezero_init), device=device))
            self.alpha_ffn = nn.Parameter(torch.tensor(float(rezero_init), device=device))
        else:
            self.alpha_attn = 1.0
            self.alpha_ffn = 1.0
        self.self_attn = MultiHeadLUTSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            lut_n_detectors=attn_lut_n_detectors,
            lut_n_anchors_per_detector=attn_lut_n_anchors_per_detector,
            projection_sampling_policy=projection_sampling_policy,
            use_pair_lut_mmul=use_pair_lut_mmul,
            normalize_scores_by_max=normalize_scores_by_max,
            score_norm_eps=score_norm_eps,
            gradient_policy=gradient_policy,
            lut_synapse_meta=lut_synapse_meta,
            device=device,
        )
        self.ffn_in = LUTLayer(
            n_inputs=embedding_dim,
            n_outputs=ffn_dim,
            n_detectors=ffn_lut_n_detectors,
            n_anchors_per_detector=ffn_lut_n_anchors_per_detector,
            weights_gradient_policy=gradient_policy,
            synapse_meta=lut_synapse_meta,
            sequence_length=1,
            device=device,
        )
        self.ffn_out = LUTLayer(
            n_inputs=ffn_dim,
            n_outputs=embedding_dim,
            n_detectors=ffn_lut_n_detectors,
            n_anchors_per_detector=ffn_lut_n_anchors_per_detector,
            weights_gradient_policy=gradient_policy,
            synapse_meta=lut_synapse_meta,
            sequence_length=1,
            device=device,
        )
        self.dropout = nn.Dropout(dropout)

    def set_external_learning_rate_hook(self, lr_hook):
        self.self_attn.set_external_learning_rate_hook(lr_hook)
        self.ffn_in.set_external_learning_rate_hook(lr_hook)
        self.ffn_out.set_external_learning_rate_hook(lr_hook)

    @staticmethod
    def _apply_lut_3d(lut_layer, x):
        # x: (B, S, N) -> (B * S, N) -> LUT expects (B * S, 1, N) -> (B, S, M).
        batch_size, seq_len, n_inputs = x.shape
        x_flat = x.reshape(batch_size * seq_len, 1, n_inputs)
        out_flat = lut_layer(x_flat).reshape(batch_size, seq_len, -1)
        return out_flat

    def forward(self, x, causal_mask):
        # x: (B, S, E), causal_mask: (1, 1, S, S).
        # Attention block with residual path.
        # attn_in: (B, S, E).
        attn_in = x
        # attn_out: (B, S, E).
        attn_out = self.self_attn(attn_in, causal_mask=causal_mask)
        # Residual result: (B, S, E).
        x = x + self.alpha_attn * self.dropout(attn_out)

        # FFN block with two explicit matrix multiplications.
        # ffn_in: (B, S, E).
        ffn_in = x
        # ffn_hidden: (B, S, F), where F = ffn_dim.
        ffn_hidden = self._apply_lut_3d(self.ffn_in, ffn_in)
        ffn_hidden = self.dropout(ffn_hidden)
        # ffn_out: (B, S, E).
        ffn_out = self._apply_lut_3d(self.ffn_out, ffn_hidden)
        # Residual result: (B, S, E).
        x = x + self.alpha_ffn * self.dropout(ffn_out)
        return x


class LUTTransformerNew(nn.Module):
    """
    A simple decoder-only Transformer sketch with trainable positional embeddings.
    """

    def __init__(
        self,
        vocab_size,
        context_size,
        num_layers=6,
        num_heads=4,
        embedding_dim=64,
        ffn_dim=64,
        dropout=0.1,
        attn_lut_n_detectors=16,
        attn_lut_n_anchors_per_detector=10,
        projection_sampling_policy: PointSamplingPolicy = None,
        use_pair_lut_mmul=False,
        ffn_lut_n_detectors=16,
        ffn_lut_n_anchors_per_detector=10,
        normalize_scores_by_max=True,
        score_norm_eps=1e-6,
        rezero_init=None,
        positional_merge_mode="concat",
        use_reverse_blocks=False,
        gradient_policy: GradientPolicy = None,
        lut_synapse_meta=SynapseMeta(),
        device=None,
    ):
        super().__init__()

        if positional_merge_mode not in ("concat", "sum"):
            raise ValueError('positional_merge_mode must be either "concat" or "sum"')
        if positional_merge_mode == "concat" and embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be divisible by 2 for token/positional concatenation")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.positional_merge_mode = positional_merge_mode
        if self.positional_merge_mode == "concat":
            self.token_embedding_dim = embedding_dim // 2
            self.positional_embedding_dim = embedding_dim // 2
        else:
            self.token_embedding_dim = embedding_dim
            self.positional_embedding_dim = embedding_dim

        if device is None:
            device = torch.device("cpu")

        self.token_embedding = nn.Embedding(vocab_size, self.token_embedding_dim, device=device)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, context_size, self.positional_embedding_dim, device=device)
        )

        self.dropout = nn.Dropout(dropout)
        self.use_reverse_blocks = use_reverse_blocks
        self.layers = nn.ModuleList(
            [
                LUTTransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attn_lut_n_detectors=attn_lut_n_detectors,
                    attn_lut_n_anchors_per_detector=attn_lut_n_anchors_per_detector,
                    projection_sampling_policy=projection_sampling_policy,
                    use_pair_lut_mmul=use_pair_lut_mmul,
                    ffn_lut_n_detectors=ffn_lut_n_detectors,
                    ffn_lut_n_anchors_per_detector=ffn_lut_n_anchors_per_detector,
                    normalize_scores_by_max=normalize_scores_by_max,
                    score_norm_eps=score_norm_eps,
                    rezero_init=rezero_init,
                    gradient_policy=gradient_policy,
                    lut_synapse_meta=lut_synapse_meta,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        # Reverse blocks that predict previous layer representations
        self.reverse_layers = None
        if self.use_reverse_blocks:
            self.reverse_layers = nn.ModuleList(
                [
                    LUTTransformerBlock(
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        dropout=dropout,
                        attn_lut_n_detectors=attn_lut_n_detectors,
                        attn_lut_n_anchors_per_detector=attn_lut_n_anchors_per_detector,
                        projection_sampling_policy=projection_sampling_policy,
                        use_pair_lut_mmul=use_pair_lut_mmul,
                        ffn_lut_n_detectors=ffn_lut_n_detectors,
                        ffn_lut_n_anchors_per_detector=ffn_lut_n_anchors_per_detector,
                        normalize_scores_by_max=normalize_scores_by_max,
                        score_norm_eps=score_norm_eps,
                        rezero_init=rezero_init,
                        gradient_policy=gradient_policy,
                        lut_synapse_meta=lut_synapse_meta,
                        device=device,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False, device=device)

    def set_external_learning_rate_hook(self, lr_hook):
        # Applies only to LUT layers; token embedding and output head are standard PyTorch layers.
        for layer in self.layers:
            layer.set_external_learning_rate_hook(lr_hook)
        if self.reverse_layers is not None:
            for reverse_layer in self.reverse_layers:
                reverse_layer.set_external_learning_rate_hook(lr_hook)

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

        # Token embeddings: (B, S) -> (B, S, token_E).
        token_x = self.token_embedding(tokens)
        # Positional slice: (1, S, pos_E) -> broadcast to (B, S, pos_E).
        pos_x = self.positional_embedding[:, :seq_len, :].expand(token_x.shape[0], -1, -1)
        if self.positional_merge_mode == "concat":
            # Concatenate embeddings: (B, S, E/2) + (B, S, E/2) -> (B, S, E).
            x = torch.cat([token_x, pos_x], dim=-1)
        else:
            # Sum embeddings (vanilla style): (B, S, E) + (B, S, E) -> (B, S, E).
            x = token_x + pos_x
        # Dropout keeps shape: (B, S, E).
        x = self.dropout(x)

        # Causal mask: (1, 1, S, S).
        mask = self._causal_mask(seq_len=seq_len, device=tokens.device)
        
        # Store intermediate representations for reverse block loss computation
        intermediate_reps = []
        if self.use_reverse_blocks:
            intermediate_reps.append(x)  # Store initial embedding
        
        # Each block keeps shape: (B, S, E) -> (B, S, E).
        for layer in self.layers:
            x = layer(x, causal_mask=mask)
            if self.use_reverse_blocks:
                intermediate_reps.append(x)  # Store after each layer
        
        # Store intermediate representations for loss computation
        if self.use_reverse_blocks:
            self._last_intermediate_reps = intermediate_reps
            self._last_causal_mask = mask
        
        # Output projection: (B, S, E) -> (B, S, vocab_size).
        logits = self.output_head(x)

        return logits
    
    def compute_internal_loss(self, loss_type="mse", reduction="mean"):
        """
        Compute internal consistency loss using reverse blocks.
        
        For each forward layer i, the corresponding reverse block tries to predict
        the input of layer i (which is the output of layer i-1) from the output of layer i.
        This encourages the network to maintain consistent representations.
        
        Args:
            loss_type: Type of loss to use. Options: "mse", "l1", "cosine".
            reduction: How to reduce the loss. Options: "mean", "sum", "none".
        
        Returns:
            Internal loss tensor (scalar if reduction="mean" or "sum", tensor otherwise).
            Returns None if reverse blocks are not enabled or no forward pass has been made.
        """
        if not self.use_reverse_blocks or not hasattr(self, '_last_intermediate_reps'):
            return None
        
        if self.reverse_layers is None:
            return None
        
        intermediate_reps = self._last_intermediate_reps
        if len(intermediate_reps) != self.num_layers + 1:
            return None
        
        mask = self._last_causal_mask
        
        total_loss = []
        # For each layer i, reverse block i predicts the input of layer i from its output
        for i in range(self.num_layers):
            # Current layer output (input to reverse block)
            current_output = intermediate_reps[i + 1]  # Output of layer i
            # Target: what the reverse block should predict (input to layer i)
            # Detach target to prevent gradients from reverse loss flowing back through forward pass
            target_input = intermediate_reps[i].detach()  # Input to layer i (detached)
            
            # Reverse block predicts previous layer representation
            predicted_input = self.reverse_layers[i](current_output, causal_mask=mask)
            
            # Compute loss between predicted and actual previous representation
            if loss_type == "mse":
                loss = torch.nn.functional.mse_loss(predicted_input, target_input, reduction=reduction)
            elif loss_type == "l1":
                loss = torch.nn.functional.l1_loss(predicted_input, target_input, reduction=reduction)
            elif loss_type == "cosine":
                # Cosine similarity loss: 1 - cosine_similarity
                predicted_flat = predicted_input.view(-1, predicted_input.shape[-1])
                target_flat = target_input.view(-1, target_input.shape[-1])
                cosine_sim = torch.nn.functional.cosine_similarity(
                    predicted_flat, target_flat, dim=-1
                )
                loss = (1.0 - cosine_sim)
                if reduction == "mean":
                    loss = loss.mean()
                elif reduction == "sum":
                    loss = loss.sum()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'mse', 'l1', or 'cosine'.")
            
            total_loss.append(loss)
        
        if not total_loss:
            return None
        
        # Combine losses from all reverse blocks
        if reduction == "none":
            return torch.stack(total_loss)
        elif reduction == "mean":
            return torch.stack(total_loss).mean()
        elif reduction == "sum":
            return torch.stack(total_loss).sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

