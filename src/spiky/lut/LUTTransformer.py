import torch
import torch.nn as nn
import torch.nn.functional as nf

from spiky.lut.LUTLayer import (
    LUTLayer,
    Conv2DLUTLayer,
    LUTSharedContext,
    GradientPolicy,
    GradientType,
    SynapseMeta
)


class LUTTransformer(nn.Module):
    def _create_single_attention(
        self, _synapse_meta, summation_dtype, _int_rescaler, seed,
        _forward_group_size, _backward_group_size, num_heads, inject_pe,
        do_normalise_weights
    ):
        p_dim = self.positional_dim if inject_pe else 0
        if isinstance(self.embedding_dim, int):
            return LUTLayer(
                n_inputs=self.embedding_dim,
                n_outputs=self.embedding_dim,
                n_detectors=self.n_detectors * num_heads,
                n_anchors_per_detector=self.n_anchors_per_detector_attention,
                sequence_length=self.context_size,
                synapse_meta=_synapse_meta,
                concatenation_product=self.concatenation_product,
                sliced_product_mode=self.sliced_product_mode,
                positional_dim=p_dim,
                use_sinusoidal_pe=p_dim > 0 and self.use_sinusoidal_pe,
                unified_pe=self.unified_pe,
                do_normalise_weights=do_normalise_weights,
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
            return Conv2DLUTLayer(
                input_shape=self.embedding_dim,
                n_anchors_per_detector=self.n_anchors_per_detector_attention,
                detectors_shape=(1, self.n_detectors * num_heads),
                output_kernel_shape=self.embedding_dim,
                sequence_length=self.context_size,
                concatenation_product=self.concatenation_product,
                sliced_product_mode=self.sliced_product_mode,
                positional_dim=p_dim,
                use_sinusoidal_pe=p_dim > 0 and self.use_sinusoidal_pe,
                unified_pe=self.unified_pe,
                do_normalise_weights=do_normalise_weights,
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
        n_detectors, n_anchors_per_detector, n_anchors_per_detector_attention=None,
        no_ffn=False, concatenation_product=True, sliced_product_mode=False,
        use_sinusoidal_pe=False, unified_pe=False, inject_pe_once=False,
        do_normalise_weights=False,
        weights_gradient_policy=None,
        relu_before_luts=False,
        device=None, _synapse_meta=SynapseMeta(),
        lut_shared_context=None, seed=None, summation_dtype=torch.float32, _int_rescaler=0.001,
        _forward_group_size=32, _backward_group_size=32, dropout=0.0,
        use_batch_norm=False, layer_norm_d=None
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
        self.concatenation_product = concatenation_product
        self.sliced_product_mode = sliced_product_mode
        self.weights_gradient_policy = weights_gradient_policy
        self.relu_before_luts = relu_before_luts
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.layer_norm_d = layer_norm_d
        self.use_sinusoidal_pe = use_sinusoidal_pe
        self.unified_pe = unified_pe
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

        self.inject_pe_once = inject_pe_once

        # Transformer layers
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer = nn.ModuleDict()

            layer['attention_lut'] = self._create_single_attention(
                _synapse_meta=_synapse_meta, summation_dtype=summation_dtype,
                _int_rescaler=_int_rescaler,
                seed=None if seed is None else seed + layer_idx * num_heads,
                _forward_group_size=_forward_group_size,
                _backward_group_size=_backward_group_size,
                num_heads=num_heads, inject_pe=not inject_pe_once or layer_idx == 0,
                do_normalise_weights=do_normalise_weights
            )

            # Dropout after attention
            layer['attention_dropout'] = nn.Dropout(dropout)
            
            # Batch normalization after attention
            if use_batch_norm:
                layer['attention_bn'] = nn.BatchNorm1d(n_embeddings, device=device)
            else:
                layer['attention_bn'] = None

            # Layer normalization after attention
            if layer_norm_d is not None:
                assert 0 < layer_norm_d < 3
                layer['attention_ln'] = nn.LayerNorm(
                    (context_size, n_embeddings) if layer_norm_d == 2 else n_embeddings, device=device
                )
            else:
                layer['attention_ln'] = None

            self.no_ffn = no_ffn
            if not no_ffn:
                ffn_lut = LUTLayer(
                    n_inputs=n_embeddings,
                    n_outputs=n_embeddings,
                    n_detectors=n_detectors,
                    n_anchors_per_detector=n_anchors_per_detector,
                    sequence_length=1,  # sequence is processed via simple reshape: [B, S, E] -> [B * S, 1, E]
                    synapse_meta=_synapse_meta,
                    do_normalise_weights=do_normalise_weights,
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

                # Dropout after FFN
                layer['ffn_dropout'] = nn.Dropout(dropout)

                # Batch normalization after FFN
                if use_batch_norm:
                    layer['ffn_bn'] = nn.BatchNorm1d(n_embeddings, device=device)
                else:
                    layer['ffn_bn'] = None

                # Layer normalization after FFN
                if layer_norm_d is not None:
                    layer['ffn_ln'] = nn.LayerNorm(
                        (context_size, n_embeddings) if layer_norm_d == 2 else n_embeddings,
                        device=device
                    )
                else:
                    layer['ffn_ln'] = None

            self.layers.append(layer)

        self.unembedder = LUTLayer(
            n_inputs=n_embeddings,
            n_outputs=vocab_size,
            n_detectors=n_detectors,
            n_anchors_per_detector=n_anchors_per_detector,
            sequence_length=1,  # sequence is processed via simple reshape: [B, S, E] -> [B * S, 1, E]
            synapse_meta=_synapse_meta,
            do_normalise_weights=do_normalise_weights,
            weights_gradient_policy=weights_gradient_policy,
            shared_context=self.lut_shared_context,
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            device=device,
            random_seed=seed,
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size
        )
        self._debug_last_forward = None

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
        if self._debug_last_forward is not None:
            self._debug_last_forward = [tokens.to(dtype=torch.float32)]

        batch_size = tokens.shape[0]
        # Token embedding: (batch_size, context_size) -> (batch_size, context_size, n_embeddings)
        z = self.token_embedder(tokens)  # (batch_size, context_size, n_embeddings)
        if self._debug_last_forward is not None:
            self._debug_last_forward.append(z.detach().clone())
        if isinstance(self.embedding_dim, int):
            non_seq_shape = (batch_size * self.context_size, 1, self.embedding_dim)
            seq_shape = (batch_size, self.context_size, self.embedding_dim)
        else:
            non_seq_shape = (batch_size * self.context_size, 1, self.embedding_dim[0] * self.embedding_dim[1])
            seq_shape = (batch_size, self.context_size, self.embedding_dim[0] * self.embedding_dim[1])

        for layer in self.layers:
            if not isinstance(self.embedding_dim, int):
                z = z.reshape((batch_size, self.context_size,) + self.embedding_dim)
            # Attention with residual connection and dropout
            if self.relu_before_luts:
                z = nf.relu(z)
            aat = layer['attention_lut'](z)
            # print(f'test: aat {aat.cpu().detach().numpy()}')
            if self._debug_last_forward is not None:
                self._debug_last_forward.append(aat.detach().clone())

            aat = layer['attention_dropout'](aat)
            if self.layer_norm_d is not None:
                aat = layer['attention_ln'](aat)
            if self.use_batch_norm:
                # Reshape for BatchNorm1d: (B, S, E) -> (B*S, E)
                aat_flat = aat.reshape(-1, aat.shape[-1])
                aat_flat = layer['attention_bn'](aat_flat)
                aat = aat_flat.reshape(aat.shape)

            z = z + aat

            if not isinstance(self.embedding_dim, int):
                z = z.reshape(batch_size, self.context_size, self.embedding_dim[0] * self.embedding_dim[1])

            if not self.no_ffn:
                # FFN with residual connection and dropout
                if self.relu_before_luts:
                    z = nf.relu(z)
                ffn_result = (layer['ffn'](z.reshape(non_seq_shape))).reshape(seq_shape)
                if self._debug_last_forward is not None:
                    self._debug_last_forward.append(ffn_result.detach().clone())
                # print(f'test: ffn_result {ffn_result}')
                ffn_result = layer['ffn_dropout'](ffn_result)

                if self.layer_norm_d is not None:
                    ffn_result = layer['ffn_ln'](ffn_result)
                if self.use_batch_norm:
                    # Reshape for BatchNorm1d: (B, S, E) -> (B*S, E)
                    ffn_result_flat = ffn_result.reshape(-1, ffn_result.shape[-1])
                    ffn_result_flat = layer['ffn_bn'](ffn_result_flat)
                    ffn_result = ffn_result_flat.reshape(ffn_result.shape)

                z = z + ffn_result

        # Unembedder: (batch_size, context_size, n_embeddings) -> (batch_size, context_size, vocab_size)
        if self.relu_before_luts:
            z = nf.relu(z)
        logits = self.unembedder(z.reshape(non_seq_shape)).reshape(batch_size, self.context_size, self.vocab_size)
        return logits

    def _reset_shared_context(self, new_context):
        for layer in self.layers:
            layer['attention_lut']._reset_shared_context(new_context)
            layer['ffn']._reset_shared_context(new_context)
        self.unembedder._reset_shared_context(new_context)

    def to(self, *args, **kwargs):
        """
        Move the module to a different device or dtype.
        
        Args:
            *args: Device or dtype (e.g., 'cuda', torch.device('cuda:0'), torch.float32)
            **kwargs: Device or dtype (e.g., device='cuda', dtype=torch.float32)
        
        Returns:
            self
        """
        # Extract device from args/kwargs first to normalize it
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            # Check if first arg is a device (not dtype)
            first_arg = args[0]
            if isinstance(first_arg, torch.device):
                device = first_arg
            elif isinstance(first_arg, str):
                # Try to parse as device; if it fails, it's likely a dtype and we'll ignore it
                try:
                    device = torch.device(first_arg)
                except (ValueError, RuntimeError):
                    # Not a valid device string, likely a dtype - keep device as None
                    pass
        
        # Normalize device if provided
        normalized_device = None
        if device is not None:
            dev = device if isinstance(device, torch.device) else torch.device(device)
            
            # Handle CUDA device index
            if dev.type == 'cuda' and dev.index is None:
                device_index = torch.cuda.current_device()
                dev = torch.device(f'cuda:{device_index}')
            
            normalized_device = dev
            self.device = dev
            
            # Move shared context to the device
            self.lut_shared_context.to_device(self.device)
        
        # Call super().to() to move all standard PyTorch modules (Embedding, Dropout, BatchNorm, LayerNorm)
        super().to(*args, **kwargs)
        
        # Explicitly call to() on all LUT layers to ensure their custom device logic runs
        # This is necessary because LUT layers have custom to() methods that update
        # their internal state (self.device, _lut_dm, etc.)
        # Use normalized device if available, otherwise pass through args/kwargs
        if normalized_device is not None:
            # Pass normalized device to LUT layers
            for layer in self.layers:
                layer['attention_lut'].to(device=normalized_device)
                if not self.no_ffn:
                    layer['ffn'].to(device=normalized_device)
            self.unembedder.to(device=normalized_device)
        else:
            # No device change, but might be dtype change - pass through args/kwargs
            for layer in self.layers:
                layer['attention_lut'].to(*args, **kwargs)
                if not self.no_ffn:
                    layer['ffn'].to(*args, **kwargs)
            self.unembedder.to(*args, **kwargs)
        
        return self

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
            attention_lut = layer['attention_lut']
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
