"""
GTLUTProduct: A module that processes two input tensors using LUTLayer with positional embeddings.

The module processes pairs of positions (i, j) from two input sequences, combining them with
positional embeddings to create triples that are processed by an inner LUTLayer.
"""

import torch
import torch.nn as nn
from spiky.lut.LUTLayer import LUTLayer, SynapseMeta, GradientPolicy, LUTSharedContext


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
        weights_gradient_policy: GradientPolicy = None,
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
            weights_gradient_policy: Gradient policy for the inner LUTLayer
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
            weights_gradient_policy=weights_gradient_policy,
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
    
    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> torch.Tensor:
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
                    pos_emb = self.positional_embeddings[pos_idx].unsqueeze(0).expand(batch_size, -1)  # [batch_size, positional_dim]
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
                
                # Process with inner LUTLayer
                lut_output = self.lut_layer(stacked_triples)  # [batch_size * n_triples, 1, n_outputs]
                lut_output = lut_output.squeeze(1)  # [batch_size * n_triples, n_outputs]
                
                # Reshape and sum over all triples for each batch item to get output[j]
                lut_output = lut_output.view(batch_size, n_triples, self.n_outputs)  # [batch_size, n_triples, n_outputs]
                output[:, j, :] = lut_output.sum(dim=1)  # [batch_size, n_outputs]
        
        return output
    
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

