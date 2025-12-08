import torch
import torch.nn as nn

from spiky.lut.LUTLayer import (
    LUTLayer,
    Conv2DLUTLayer,
    LUTSharedContext,
    GradientPolicy,
    GradientType,
    MultiLUT,
    SynapseMeta
)


class LUTTransformer(nn.Module):
    def _create_single_attention(
        self, _synapse_meta, summation_dtype, _int_rescaler, seed,
        _forward_group_size, _backward_group_size, num_heads
    ):
        if isinstance(self.embedding_dim, int):
            return LUTLayer(
                n_inputs=self.embedding_dim,
                n_outputs=self.embedding_dim,
                n_detectors=self.n_detectors * num_heads,
                n_anchors_per_detector=self.n_anchors_per_detector,
                sequence_length=self.context_size,
                synapse_meta=_synapse_meta,
                positional_dim=self.positional_dim,
                weights_gradient_policy=self.weights_gradient_policy,
                shared_context=self.lut_shared_context,
                summation_dtype=summation_dtype,
                _int_rescaler=_int_rescaler,
                device=self.device,
                random_seed=seed,
                _forward_group_size=_forward_group_size,
                _backward_group_size=_backward_group_size
            )
        else:
            # right now this branch is needed only for tests
            assert num_heads == 1
            return Conv2DLUTLayer(
                input_shape=self.embedding_dim,
                n_anchors_per_detector=self.n_anchors_per_detector,
                detectors_shape=(1, self.n_detectors),
                output_kernel_shape=self.embedding_dim,
                sequence_length=self.context_size,
                positional_dim=self.positional_dim,
                weights_gradient_policy=self.weights_gradient_policy,
                receptive_field_shape=self.embedding_dim,
                receptive_field_stride_shape=self.embedding_dim,
                synapse_meta=_synapse_meta,
                shared_context=self.lut_shared_context,
                summation_dtype=summation_dtype,
                random_seed=seed,
                _int_rescaler=_int_rescaler,
                device=self.device,
                _forward_group_size=_forward_group_size,
                _backward_group_size=_backward_group_size
            )

    def __init__(
        self, vocab_size, embedding_dim, context_size,
        positional_dim, num_layers, num_heads,
        n_detectors, n_anchors_per_detector, weights_gradient_policy=None,
        device=None, _synapse_meta=SynapseMeta(), _use_multi_lut=False,
        lut_shared_context=None, seed=None, summation_dtype=torch.float32, _int_rescaler=0.001,
        _forward_group_size=32, _backward_group_size=32
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
        self.weights_gradient_policy = weights_gradient_policy
        if device is None:
            device = torch.device('cpu')

        self.device = device

        if lut_shared_context is None:
            self.lut_shared_context = LUTSharedContext()
            self.lut_shared_context.to_device(device)
        else:
            self.lut_shared_context = lut_shared_context

        if isinstance(embedding_dim, int):
            n_embeddings = embedding_dim
        else:
            assert len(embedding_dim) == 2
            n_embeddings = embedding_dim[0] * embedding_dim[1]

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
            if _use_multi_lut:
                heads = []
                for head_idx in range(num_heads):
                    attention_lut = self._create_single_attention(
                        _synapse_meta=_synapse_meta, summation_dtype=summation_dtype,
                        _int_rescaler=_int_rescaler,
                        seed=None if seed is None else seed + layer_idx * num_heads + head_idx,
                        _forward_group_size=_forward_group_size,
                        _backward_group_size=_backward_group_size,
                        num_heads=1
                    )
                    heads.append(attention_lut)
                layer['attention_lut'] = MultiLUT(heads)
            else:
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
                weights_gradient_policy=weights_gradient_policy,
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
            weights_gradient_policy=weights_gradient_policy,
            shared_context=self.lut_shared_context,
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            device=device,
            random_seed=seed,
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size
        )

    def set_external_learning_rate_hook(self, lr_hook):
        # Set hooks for all LUT layers
        for layer in self.layers:
            layer['attention_lut'].set_external_learning_rate_hook(lr_hook)
            layer['ffn'].set_external_learning_rate_hook(lr_hook)
        self.unembedder.set_external_learning_rate_hook(lr_hook)

    def forward(self, tokens):
        """
        Forward pass.

        Args:
            tokens: (batch_size, context_size) tensor of token indices

        Returns:
            logits: (batch_size, context_size, vocab_size) tensor of logits
        """
        batch_size = tokens.shape[0]
        # Token embedding: (batch_size, context_size) -> (batch_size, context_size, n_embeddings)
        z = self.token_embedder(tokens)  # (batch_size, context_size, n_embeddings)

        if isinstance(self.embedding_dim, int):
            non_seq_shape = (batch_size * self.context_size, 1, self.embedding_dim)
            seq_shape = (batch_size, self.context_size, self.embedding_dim)
        else:
            non_seq_shape = (batch_size * self.context_size, 1, self.embedding_dim[0] * self.embedding_dim[1])
            seq_shape = (batch_size, self.context_size, self.embedding_dim[0] * self.embedding_dim[1])

        for layer in self.layers:
            if not isinstance(self.embedding_dim, int):
                z = z.reshape((batch_size, self.context_size,) + self.embedding_dim)
            z = z + layer['attention_lut'](z)
            if not isinstance(self.embedding_dim, int):
                z = z.reshape(batch_size, self.context_size, self.embedding_dim[0] * self.embedding_dim[1])
            z = z + (layer['ffn'](z.reshape(non_seq_shape))).reshape(seq_shape)

        # Unembedder: (batch_size, context_size, n_embeddings) -> (batch_size, context_size, vocab_size)
        logits = self.unembedder(z.reshape(non_seq_shape)).reshape(batch_size, self.context_size, self.vocab_size)
        return logits

    def _reset_shared_context(self, new_context):
        for layer in self.layers:
            layer['attention_lut']._reset_shared_context(new_context)
            layer['ffn']._reset_shared_context(new_context)
        self.unembedder._reset_shared_context(new_context)

    def get_profile_statistics(self) -> str:
        """
        Get aggregated profiling statistics from all LUT layers in the transformer.
        Only includes lut::runtime operations, grouped and averaged by component type.
        
        Returns:
            String with aggregated profiling statistics in the format:
            - Average Attention metrics (averaged across all attention layers)
            - Average FFN metrics (averaged across all FFN layers)
            - Unembedder metrics (single layer)
            Format: operation_name: total_time ms / total_count = avg_time ms
        """
        import re
        from collections import defaultdict
        
        # Collect LUT layers by component type
        attention_luts = []
        ffn_luts = []
        unembedder_lut = self.unembedder
        
        # Collect from transformer layers
        for layer in self.layers:
            # Handle attention_lut (can be LUTLayer or MultiLUT)
            attention_lut = layer['attention_lut']
            if isinstance(attention_lut, MultiLUT):
                attention_luts.extend(attention_lut.luts)
            else:
                attention_luts.append(attention_lut)
            
            # FFN is always a single LUTLayer
            ffn_luts.append(layer['ffn'])
        
        # Aggregate statistics by component type
        attention_stats = defaultdict(lambda: {'total_time': 0.0, 'total_count': 0})
        ffn_stats = defaultdict(lambda: {'total_time': 0.0, 'total_count': 0})
        unembedder_stats = defaultdict(lambda: {'total_time': 0.0, 'total_count': 0})
        
        def parse_and_aggregate(profiling_stats, stats_dict):
            """Parse profiling stats and aggregate into stats_dict"""
            for line in profiling_stats.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Only process lut::runtime lines
                if 'lut::runtime' not in line:
                    continue
                
                # Parse: operation_name: total_time ms / count = avg_time ms
                # Example: lut::runtime::forward_step: 11750.9 ms / 1164 = 10.0952 ms
                match = re.match(r'^([^:]+(?:::[^:]+)*):\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+=\s+([\d.-]+)\s+ms$', line)
                if match:
                    op_name = match.group(1)
                    total_time = float(match.group(2))
                    count = int(match.group(3))
                    
                    stats_dict[op_name]['total_time'] += total_time
                    stats_dict[op_name]['total_count'] += count
        
        # Parse statistics from attention layers
        for lut in attention_luts:
            parse_and_aggregate(lut.get_profiling_stats(), attention_stats)
        
        # Parse statistics from FFN layers
        for lut in ffn_luts:
            parse_and_aggregate(lut.get_profiling_stats(), ffn_stats)
        
        # Parse statistics from unembedder
        parse_and_aggregate(unembedder_lut.get_profiling_stats(), unembedder_stats)
        
        # Format output
        result_lines = []
        
        # Average Attention metrics
        if attention_luts:
            result_lines.append("Average Attention:")
            n_layers = len(attention_luts)
            # Sort by average total_time (total_time / n_layers) descending
            sorted_attention = sorted(
                attention_stats.items(),
                key=lambda x: x[1]['total_time'] / n_layers,
                reverse=True
            )
            for op_name, stats in sorted_attention:
                total_time = stats['total_time']
                total_count = stats['total_count']
                
                if total_count > 0:
                    avg_time = total_time / total_count
                    avg_total_time = total_time / n_layers
                    avg_count = total_count / n_layers
                    result_lines.append(f"  {op_name}: {avg_total_time:.6g} ms / {avg_count:.1f} = {avg_time:.6g} ms")
                else:
                    avg_total_time = total_time / n_layers
                    result_lines.append(f"  {op_name}: {avg_total_time:.6g} ms / 0 = -nan ms")
            result_lines.append("")
        
        # Average FFN metrics
        if ffn_luts:
            result_lines.append("Average FFN:")
            n_layers = len(ffn_luts)
            # Sort by average total_time (total_time / n_layers) descending
            sorted_ffn = sorted(
                ffn_stats.items(),
                key=lambda x: x[1]['total_time'] / n_layers,
                reverse=True
            )
            for op_name, stats in sorted_ffn:
                total_time = stats['total_time']
                total_count = stats['total_count']
                
                if total_count > 0:
                    avg_time = total_time / total_count
                    avg_total_time = total_time / n_layers
                    avg_count = total_count / n_layers
                    result_lines.append(f"  {op_name}: {avg_total_time:.6g} ms / {avg_count:.1f} = {avg_time:.6g} ms")
                else:
                    avg_total_time = total_time / n_layers
                    result_lines.append(f"  {op_name}: {avg_total_time:.6g} ms / 0 = -nan ms")
            result_lines.append("")
        
        # Unembedder metrics
        result_lines.append("Unembedder:")
        # Sort by total_time descending
        sorted_unembedder = sorted(
            unembedder_stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        for op_name, stats in sorted_unembedder:
            total_time = stats['total_time']
            total_count = stats['total_count']
            
            if total_count > 0:
                avg_time = total_time / total_count
                result_lines.append(f"  {op_name}: {total_time:.6g} ms / {total_count} = {avg_time:.6g} ms")
            else:
                result_lines.append(f"  {op_name}: {total_time:.6g} ms / {total_count} = -nan ms")
        
        return '\n'.join(result_lines)
