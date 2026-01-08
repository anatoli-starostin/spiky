"""
GTLUTProduct: A module that processes two input tensors using LUTLayer with positional embeddings.

The module processes pairs of positions (i, j) from two input sequences, combining them with
positional embeddings to create triples that are processed by an inner LUTLayer.
"""

import torch
import torch.nn as nn
from spiky.lut.LUTLayer import LUTLayer, SynapseMeta, GradientPolicy, GradientType, LUTSharedContext


class GTLUTProduct(nn.Module):
    """
    A module that processes two input tensors using LUTLayer with positional embeddings.
    
    For each output position j, processes all pairs (i, j) where i < j, combining
    input_1[i], input_2[j], and positional_embedding[j - i] into triples that are
    processed by an inner LUTLayer.
    """
    
    def __init__(
        self,
        n_inputs_1: int,
        n_inputs_2: int,
        positional_dim: int,
        n_outputs: int,
        sequence_length: int,
        sliced_mode: bool = False,
        n_detectors: int = 8,
        n_anchors_per_detector: int = 6,
        synapse_meta: SynapseMeta = None,
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
        random_seed=None,
        device=None
    ):
        """
        Initialize GTLUTProduct.
        
        Args:
            n_inputs_1: Number of inputs for the first tensor
            n_inputs_2: Number of inputs for the second tensor
            positional_dim: Dimension of positional embeddings
            n_outputs: Number of outputs
            sequence_length: Sequence length (must match input sequence length)
            sliced_mode: If True, merge inputs element-wise (interleave) instead of concatenating.
                        Requires n_inputs_1 == n_inputs_2 == positional_dim.
            n_detectors: Number of detectors for the inner LUTLayer
            n_anchors_per_detector: Number of anchor pairs per detector
            synapse_meta: Synapse metadata for the inner LUTLayer
            shared_context: Shared context for the inner LUTLayer
            summation_dtype: Data type for summations
            random_seed: Random seed for initialization
            device: Device to use
        """
        super().__init__()
        
        assert sequence_length > 1, f"sequence_length must be > 1, got {sequence_length}"
        
        self.n_inputs_1 = n_inputs_1
        self.n_inputs_2 = n_inputs_2
        self.positional_dim = positional_dim
        self.n_outputs = n_outputs
        self.sequence_length = sequence_length
        self.sliced_mode = sliced_mode
        
        # Validate sliced mode requirements
        if sliced_mode:
            assert n_inputs_1 == n_inputs_2 == positional_dim, \
                f"In sliced_mode, n_inputs_1 ({n_inputs_1}), n_inputs_2 ({n_inputs_2}), and positional_dim ({positional_dim}) must be equal"
            assert positional_dim > 0, "positional_dim must be > 0 in sliced_mode"
        else:
            assert positional_dim >= 0, "positional_dim must be >= 0"
        
        # Inner LUTLayer processes merged inputs
        if sliced_mode:
            # Element-wise merge: interleave [input_1[0], input_2[0], pos_emb[0], input_1[1], ...]
            n_lut_inputs = n_inputs_1 * 3  # All three have the same dimension
        else:
            # Concatenate: [input_1, input_2, positional_embedding]
            n_lut_inputs = n_inputs_1 + n_inputs_2 + positional_dim
        
        if synapse_meta is None:
            synapse_meta = SynapseMeta()
        
        # Create inner LUTLayer with sequence_length=1
        self.lut_layer = LUTLayer(
            n_inputs=n_lut_inputs,
            n_anchors_per_detector=n_anchors_per_detector,
            n_detectors=n_detectors,
            n_outputs=n_outputs,
            sequence_length=1,
            synapse_meta=synapse_meta,
            positional_dim=None,  # No positional embeddings in inner LUTLayer
            weights_gradient_policy=GradientPolicy(GradientType.Dense),
            shared_context=shared_context,
            summation_dtype=summation_dtype,
            random_seed=random_seed,
            device=device
        )
        
        # Initialize positional embeddings: [sequence_length - 1, positional_dim]
        if positional_dim > 0:
            pos_emb = torch.empty(
                sequence_length - 1,
                positional_dim,
                dtype=torch.float32,
                device=device if device is not None else torch.device("cpu")
            )
            # Initialize with random values in [-1, 1]
            pos_emb.uniform_(-1.0, 1.0)
            self.positional_embeddings = nn.Parameter(pos_emb)
        else:
            self.positional_embeddings = None
        
        if device is not None:
            self.to(device=device)
    
    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_1: First input tensor of shape [batch_size, sequence_length, n_inputs_1]
            input_2: Second input tensor of shape [batch_size, sequence_length, n_inputs_2]
            
        Returns:
            Output tensor of shape [batch_size, sequence_length, n_outputs]
        """
        batch_size = input_1.shape[0]
        sequence_length = input_1.shape[1]

        if input_2 is None:
            input_2 = input_1
        
        # Validate input shapes
        assert sequence_length == self.sequence_length, \
            f"Input sequence_length {sequence_length} does not match constructor sequence_length {self.sequence_length}"
        assert input_1.shape == (batch_size, sequence_length, self.n_inputs_1), \
            f"input_1 shape mismatch: expected {(batch_size, sequence_length, self.n_inputs_1)}, got {input_1.shape}"
        assert input_2.shape == (batch_size, sequence_length, self.n_inputs_2), \
            f"input_2 shape mismatch: expected {(batch_size, sequence_length, self.n_inputs_2)}, got {input_2.shape}"
        
        # Initialize output tensor
        output = torch.zeros(
            batch_size, sequence_length, self.n_outputs,
            dtype=torch.float32, device=input_1.device
        )
        
        # Process each output position j
        for j in range(sequence_length):
            # Collect all triples (input_1[i], input_2[j], positional_embedding[j - i])
            triples = []
            for i in range(0, j):
                # Get positional embedding if available
                if self.positional_embeddings is not None:
                    pos_idx = j - i
                    # pos_idx is guaranteed to be < sequence_length - 1 (in range [1, sequence_length - 2])
                    pos_emb = self.positional_embeddings[pos_idx - 1].unsqueeze(0).repeat(batch_size, 1)  # [batch_size, positional_dim]
                else:
                    pos_emb = None
                
                # Merge input_1[i], input_2[j], and positional_embedding[j - i]
                if self.sliced_mode:
                    # Element-wise merge: interleave [input_1[0], input_2[0], pos_emb[0], input_1[1], input_2[1], pos_emb[1], ...]
                    input_1_slice = input_1[:, i, :]  # [batch_size, n_inputs_1]
                    input_2_slice = input_2[:, j, :]  # [batch_size, n_inputs_2]
                    # Stack along dim=2 to get [batch_size, n_inputs_1, 3], then reshape to interleave
                    stacked = torch.stack([input_1_slice, input_2_slice, pos_emb], dim=2)  # [batch_size, n_inputs_1, 3]
                    triple = stacked.reshape(batch_size, -1)  # [batch_size, n_inputs_1 * 3]
                else:
                    # Concatenate input_1[i], input_2[j], and positional_embedding[j - i] (if available)
                    if pos_emb is not None:
                        triple = torch.cat([
                            input_1[:, i, :],  # [batch_size, n_inputs_1]
                            input_2[:, j, :],  # [batch_size, n_inputs_2]
                            pos_emb  # [batch_size, positional_dim]
                        ], dim=1)  # [batch_size, n_inputs_1 + n_inputs_2 + positional_dim]
                    else:
                        # No positional embeddings (positional_dim == 0)
                        triple = torch.cat([
                            input_1[:, i, :],  # [batch_size, n_inputs_1]
                            input_2[:, j, :],  # [batch_size, n_inputs_2]
                        ], dim=1)  # [batch_size, n_inputs_1 + n_inputs_2]
                triples.append(triple)
            
            if len(triples) > 0:
                # Stack all triples: [batch_size, n_triples, n_lut_inputs]
                stacked_triples = torch.stack(triples, dim=1)  # [batch_size, n_triples, n_lut_inputs]
                n_triples = stacked_triples.shape[1]
                
                # Reshape for LUTLayer: [batch_size * n_triples, 1, n_lut_inputs]
                stacked_triples = stacked_triples.view(batch_size * n_triples, 1, -1)

                inp = stacked_triples.contiguous()

                # Process with inner LUTLayer
                lut_output = self.lut_layer(inp)  # [batch_size * n_triples, 1, n_outputs]
                lut_output = lut_output.squeeze(1)  # [batch_size * n_triples, n_outputs]
                
                # Reshape and sum over all triples for each batch item to get output[j]
                lut_output = lut_output.view(batch_size, n_triples, self.n_outputs)  # [batch_size, n_triples, n_outputs]
                output[:, j, :] = lut_output.sum(dim=1)  # [batch_size, n_outputs]
        
        return output.contiguous()
    
    def to(self, *args, **kwargs):
        """Move module to device/dtype."""
        result = super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        
        if device is not None:
            dev = torch.device(device)
            if self.lut_layer is not None:
                self.lut_layer.to(device=dev)
            if self.positional_embeddings is not None:
                self.positional_embeddings = self.positional_embeddings.to(device=dev)
        
        return result

#
# class GTLUTProductMultiHead(nn.Module):
#     """
#     A module that runs multiple GTLUTProduct instances in parallel (similar to MultiLUT).
#     All heads must have the same input/output shapes and sequence_length.
#     Returns the sum of outputs from all heads.
#     """
#
#     def __init__(self, heads):
#         """
#         Initialize GTLUTProductMultiHead.
#
#         Args:
#             heads: List of GTLUTProduct instances (one per head)
#         """
#         super().__init__()
#
#         if len(heads) < 1:
#             raise ValueError("GTLUTProductMultiHead requires at least one head")
#
#         # Validate all heads have compatible shapes
#         first_head = heads[0]
#         n_inputs_1 = first_head.n_inputs_1
#         n_inputs_2 = first_head.n_inputs_2
#         n_outputs = first_head.n_outputs
#         sequence_length = first_head.sequence_length
#         sliced_mode = first_head.sliced_mode
#
#         for i, head in enumerate(heads):
#             if not isinstance(head, GTLUTProduct):
#                 raise ValueError(f"head {i} is not an instance of GTLUTProduct")
#             if head.n_inputs_1 != n_inputs_1 or head.n_inputs_2 != n_inputs_2:
#                 raise ValueError(f"head {i} has incompatible input dimensions")
#             if head.n_outputs != n_outputs:
#                 raise ValueError(f"head {i} has incompatible output dimensions")
#             if head.sequence_length != sequence_length:
#                 raise ValueError(f"head {i} has incompatible sequence_length")
#             if head.sliced_mode != sliced_mode:
#                 raise ValueError(f"head {i} has incompatible sliced_mode")
#
#         self.heads = nn.ModuleList(heads)
#         self.n_inputs_1 = n_inputs_1
#         self.n_inputs_2 = n_inputs_2
#         self.n_outputs = n_outputs
#         self.sequence_length = sequence_length
#         self.sliced_mode = sliced_mode
#
#     def forward(self, input_1: torch.Tensor, input_2: torch.Tensor = None) -> torch.Tensor:
#         """
#         Forward pass that runs all heads and sums their outputs.
#
#         Args:
#             input_1: First input tensor of shape [batch_size, sequence_length, n_inputs_1]
#             input_2: Second input tensor of shape [batch_size, sequence_length, n_inputs_2]
#
#         Returns:
#             Output tensor of shape [batch_size, sequence_length, n_outputs]
#         """
#         # Sum outputs from all heads
#         output = None
#         for head in self.heads:
#             head_output = head(input_1, input_2)
#             if output is None:
#                 output = head_output
#             else:
#                 output = output + head_output
#         return output


class GTLUTProductTransformer(nn.Module):
    def _create_single_attention(
        self, _synapse_meta, summation_dtype, _int_rescaler, seed,
        _forward_group_size, _backward_group_size, num_heads
    ):
        """
        Create attention layer(s).
        creates a single GTLUTProduct with all heads combined.
        """
        # if use_multi_lut:
        #     # Create separate GTLUTProduct for each head (matching MultiLUT structure)
        #     heads = []
        #     for head_idx in range(num_heads):
        #         head = GTLUTProduct(
        #             n_inputs_1=self.embedding_dim,
        #             n_inputs_2=self.embedding_dim,
        #             positional_dim=self.positional_dim,
        #             n_outputs=self.embedding_dim,
        #             sequence_length=self.context_size,
        #             sliced_mode=self.sliced_mode,
        #             n_detectors=self.n_detectors,
        #             n_anchors_per_detector=self.n_anchors_per_detector_attention,
        #             synapse_meta=_synapse_meta,
        #             shared_context=self.lut_shared_context,
        #             summation_dtype=summation_dtype,
        #             random_seed=None if seed is None else seed + head_idx,
        #             device=self.device
        #         )
        #         heads.append(head)
        #     return GTLUTProductMultiHead(heads)
        # else:
        # Single GTLUTProduct with all heads combined
        return GTLUTProduct(
            n_inputs_1=self.embedding_dim,
            n_inputs_2=self.embedding_dim,
            positional_dim=self.positional_dim,
            n_outputs=self.embedding_dim,
            sequence_length=self.context_size,
            sliced_mode=self.sliced_mode,
            n_detectors=self.n_detectors * num_heads,
            n_anchors_per_detector=self.n_anchors_per_detector_attention,
            synapse_meta=_synapse_meta,
            shared_context=self.lut_shared_context,
            summation_dtype=summation_dtype,
            random_seed=seed,
            device=self.device
        )

    def __init__(
        self, vocab_size, embedding_dim, context_size,
        positional_dim, num_layers, num_heads,
        n_detectors, n_anchors_per_detector,
        device=None, _synapse_meta=SynapseMeta(),
        lut_shared_context=None, seed=None, summation_dtype=torch.float32, _int_rescaler=0.001,
        _forward_group_size=32, _backward_group_size=32,
        n_anchors_per_detector_attention=None, sliced_mode=False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.positional_dim = positional_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_detectors = n_detectors
        self.n_anchors_per_detector = n_anchors_per_detector
        # If not specified, use the same value as n_anchors_per_detector for backward compatibility
        self.n_anchors_per_detector_attention = n_anchors_per_detector_attention if n_anchors_per_detector_attention is not None else n_anchors_per_detector
        self.sliced_mode = sliced_mode
        if device is None:
            device = torch.device('cpu')

        self.device = device

        if lut_shared_context is None:
            self.lut_shared_context = LUTSharedContext()
            self.lut_shared_context.to_device(device)
        else:
            self.lut_shared_context = lut_shared_context

        n_embeddings = embedding_dim

        self.token_embedder = nn.Embedding(vocab_size, n_embeddings, device=device)
        self.token_embedder.weight.requires_grad_(False)
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            w = 2 * torch.rand(self.token_embedder.weight.shape, generator=gen, device=device) - 1.0
            self.token_embedder.weight.copy_(w)
        else:
            nn.init.uniform_(self.token_embedder.weight, -1.0, 1.0)

        # Transformer layers
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer = nn.ModuleDict()

            # Attention heads
            assert isinstance(embedding_dim, int)
            layer['attention_lut'] = self._create_single_attention(
                _synapse_meta=_synapse_meta, summation_dtype=summation_dtype,
                _int_rescaler=_int_rescaler,
                seed=None if seed is None else seed + layer_idx * num_heads,
                _forward_group_size=_forward_group_size,
                _backward_group_size=_backward_group_size,
                num_heads=num_heads
            )

            ffn_lut = LUTLayer(
                n_inputs=n_embeddings,
                n_outputs=n_embeddings,
                n_detectors=n_detectors,
                n_anchors_per_detector=n_anchors_per_detector,
                sequence_length=1,  # sequence is processed via simple reshape: [B, S, E] -> [B * S, 1, E]
                synapse_meta=_synapse_meta,
                weights_gradient_policy=GradientPolicy(GradientType.Dense),
                shared_context=self.lut_shared_context,
                summation_dtype=summation_dtype,
                _int_rescaler=_int_rescaler,
                device=device,
                random_seed=None if seed is None else seed + layer_idx * num_heads + num_heads,
                _forward_group_size=_forward_group_size,
                _backward_group_size=_backward_group_size
            )
            layer['ffn'] = ffn_lut
            self.layers.append(layer)

        self.unembedder = LUTLayer(
            n_inputs=n_embeddings,
            n_outputs=vocab_size,
            n_detectors=n_detectors,
            n_anchors_per_detector=n_anchors_per_detector,
            sequence_length=1,  # sequence is processed via simple reshape: [B, S, E] -> [B * S, 1, E]
            synapse_meta=_synapse_meta,
            weights_gradient_policy=GradientPolicy(GradientType.Dense),
            shared_context=self.lut_shared_context,
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            device=device,
            random_seed=seed,
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size
        )
        self._debug_last_forward = None

    def forward(self, tokens):
        """
        Forward pass.

        Args:
            tokens: (batch_size, context_size) tensor of token indices

        Returns:
            logits: (batch_size, context_size, vocab_size) tensor of logits
        """
        if self._debug_last_forward is not None:
            self._debug_last_forward = []

        batch_size = tokens.shape[0]
        # Token embedding: (batch_size, context_size) -> (batch_size, context_size, n_embeddings)
        z = self.token_embedder(tokens)  # (batch_size, context_size, n_embeddings)

        non_seq_shape = (batch_size * self.context_size, 1, self.embedding_dim)
        seq_shape = (batch_size, self.context_size, self.embedding_dim)
        if self._debug_last_forward is not None:
            self._debug_last_forward.append(z.detach().clone())

        for layer in self.layers:
            # print(f'gt: z {z}')
            attention_output = layer['attention_lut'](z)
            # print(f'gt: z after attention {attention_output.cpu().detach().numpy()}')
            if self._debug_last_forward is not None:
                self._debug_last_forward.append(attention_output.detach().clone())
            z = z + attention_output
            ffn_output = (layer['ffn'](z.reshape(non_seq_shape))).reshape(seq_shape)
            if self._debug_last_forward is not None:
                self._debug_last_forward.append(ffn_output.detach().clone())
            # print(f'gt: ffn_output {ffn_output}')
            z = z + ffn_output

        # Unembedder: (batch_size, context_size, n_embeddings) -> (batch_size, context_size, vocab_size)
        logits = self.unembedder(z.reshape(non_seq_shape)).reshape(batch_size, self.context_size, self.vocab_size)
        return logits
