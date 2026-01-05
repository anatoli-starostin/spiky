import torch
import torch.nn as nn
from typing import List, Dict, Tuple, AnyStr
from dataclasses import dataclass
from enum import Enum

from spiky_cuda import LUTDataManagerF, LUTDataManagerI
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper
from spiky.util.chunk_of_connections import ChunkOfConnections
from spiky.util.torch_utils import DenseToSparseConverter


class GradientType(Enum):
    """Type of gradient handling for weight gradients in LUT layers."""
    Sparse = 0
    Dense = 1
    Internal = 2


@dataclass(frozen=True)
class GradientPolicy:
    """Policy for handling weight gradients in LUT layers."""
    type: GradientType
    normalized: bool = False

    def __post_init__(self):
        if self.normalized and self.type == GradientType.Internal:
            raise ValueError("normalized cannot be combined with Internal gradient policy")


@dataclass(frozen=True, order=True)
class SynapseMeta:
    learning_rate: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    initial_noise_level: float = 0.0
    initial_weight: float = 0.0

    def __post_init__(self):
        assert 0.0 <= self.learning_rate <= 1.0
        assert self.min_weight <= self.initial_weight <= self.max_weight


class LUTSharedContext(object):
    def __init__(self, do_asserts=False):
        self._weight_gradients_buffers = []
        self._densify_indices_buffers = []
        self._densify_values_buffers = []
        self._dense_to_sparse_converters = []
        self._cuda_streams = []  # List of lists: each element is [stream0, stream1, stream2]
        self._cuda_stream_handles = []  # List of tensors: each tensor contains 3 int64 handles
        self._device = None
        self._do_asserts = do_asserts

    def _ensure_buffer(self, buffers_list, multi_id, numel, dtype, device):
        if self._device is None:
            raise RuntimeError('LUTSharedContext device is None, call to_device(...) first')

        if self._device != device:
            raise RuntimeError(f'LUTSharedContext is on {self._device}, but trying to get buffer on {device}')

        # Ensure the list is large enough
        while len(buffers_list) <= multi_id:
            buffers_list.append(None)

        buffer = buffers_list[multi_id]
        if (
            buffer is None or
            buffer.numel() < numel
        ):
            buffer = torch.zeros([numel], dtype=dtype, device=self._device, requires_grad=False)
            buffers_list[multi_id] = buffer
        return buffer

    def get_weight_gradients_buffer(self, numel, device, multi_id=0):
        buf = self._ensure_buffer(self._weight_gradients_buffers, multi_id, numel, torch.float32, device)
        if self._do_asserts:
            assert torch.count_nonzero(buf) == 0
        return buf[:numel]

    def get_dense_to_sparse_converter(self, multi_id=0):
        # Ensure the list is large enough
        while len(self._dense_to_sparse_converters) <= multi_id:
            self._dense_to_sparse_converters.append(None)

        if self._dense_to_sparse_converters[multi_id] is None:
            self._dense_to_sparse_converters[multi_id] = DenseToSparseConverter()
        return self._dense_to_sparse_converters[multi_id]

    def get_densify_buffers(self, numel, device, multi_id=0):
        indices = self._ensure_buffer(self._densify_indices_buffers, multi_id, numel, torch.int64, device)
        values = self._ensure_buffer(self._densify_values_buffers, multi_id, numel, torch.float32, device)
        return indices[:numel], values[:numel]

    def get_cuda_streams(self, device, multi_id=0):
        """
        Get CUDA stream handles tensor for the given multi_id.
        Returns a tensor with 3 int64 handles, or None if on CPU.
        """
        if device.type != 'cuda':
            return None

        # Ensure the lists are large enough
        while len(self._cuda_streams) <= multi_id:
            self._cuda_streams.append([None, None, None])
        while len(self._cuda_stream_handles) <= multi_id:
            self._cuda_stream_handles.append(None)

        # Create streams if they don't exist or if device changed
        if (
            self._cuda_streams[multi_id][0] is None or
            self._cuda_streams[multi_id][0].device != device
        ):
            self._cuda_streams[multi_id][0] = torch.cuda.Stream(device=device)
            self._cuda_streams[multi_id][1] = torch.cuda.Stream(device=device)
            self._cuda_streams[multi_id][2] = torch.cuda.Stream(device=device)

            # Create tensor with stream handles
            handles = torch.zeros([3], dtype=torch.int64, device=torch.device('cpu'))
            # cuda_stream property returns the handle as an integer
            handles[0] = self._cuda_streams[multi_id][0].cuda_stream
            handles[1] = self._cuda_streams[multi_id][1].cuda_stream
            handles[2] = self._cuda_streams[multi_id][2].cuda_stream
            self._cuda_stream_handles[multi_id] = handles

        return self._cuda_stream_handles[multi_id]

    def get_cuda_stream(self, device, multi_id=0, stream_index=0):
        """
        Get a single CUDA stream for the given multi_id and stream_index.
        
        Args:
            device: The device (torch.device) to get the stream for
            multi_id: The multi_id index (default: 0)
            stream_index: Which stream to get (0, 1, or 2, default: 0)
            
        Returns:
            torch.cuda.Stream object, or None if on CPU or stream_index is invalid
        """
        if device.type != 'cuda':
            return None
        
        assert 0 <= stream_index < 3, "stream_index must be between 0 and 2"
        
        # Ensure streams are created by calling get_cuda_streams
        self.get_cuda_streams(device, multi_id)
        
        # Ensure the streams list is large enough
        if len(self._cuda_streams) <= multi_id:
            return None
        
        return self._cuda_streams[multi_id][stream_index]

    def to_device(self, device):
        dev = device if isinstance(device, torch.device) else torch.device(device)

        if dev.type == 'cuda' and dev.index is None:
            device_index = torch.cuda.current_device()
            dev = torch.device(f'cuda:{device_index}')

        self._device = dev
        for i, buf in enumerate(self._weight_gradients_buffers):
            if buf is not None:
                self._weight_gradients_buffers[i] = buf.to(device=dev)
        for i, buf in enumerate(self._densify_indices_buffers):
            if buf is not None:
                self._densify_indices_buffers[i] = buf.to(device=dev)
        for i, buf in enumerate(self._densify_values_buffers):
            if buf is not None:
                self._densify_values_buffers[i] = buf.to(device=dev)
        # Streams need to be recreated on new device
        if dev.type == 'cuda':
            for i in range(len(self._cuda_streams)):
                if self._cuda_streams[i][0] is not None:
                    self._cuda_streams[i][0] = torch.cuda.Stream(device=dev)
                    self._cuda_streams[i][1] = torch.cuda.Stream(device=dev)
                    self._cuda_streams[i][2] = torch.cuda.Stream(device=dev)
                    handles = torch.zeros([3], dtype=torch.int64, device=torch.device('cpu'))
                    handles[0] = self._cuda_streams[i][0].cuda_stream
                    handles[1] = self._cuda_streams[i][1].cuda_stream
                    handles[2] = self._cuda_streams[i][2].cuda_stream
                    self._cuda_stream_handles[i] = handles
        else:
            # Clear streams when moving to CPU
            for i in range(len(self._cuda_streams)):
                self._cuda_streams[i] = [None, None, None]
                self._cuda_stream_handles[i] = None
        return self


class LUTLayerBasic(nn.Module):
    @staticmethod
    def n_lut_channels(n_anchors_per_detector, sequence_length, concatentation_product=True):
        if concatentation_product:
            return 1 << (n_anchors_per_detector * (2 if (sequence_length > 1) else 1))
        else:
            return 1 << n_anchors_per_detector

    def _initialize_positional_embeddings(self):
        # Handle positional embeddings
        if self._sequence_length > 1:
            assert self._positional_dim is not None, "positional_dim must be provided when sequence_length > 1"
            if self._positional_dim > 0:
                if self._use_sinusoidal_pe:
                    # position = torch.arange(self._sequence_length - 1, device=self.device).float().unsqueeze(1)
                    # inv_freq = torch.exp(
                    #     -torch.arange(0, self._positional_dim, 2, device=self.device).float() * (torch.log(torch.tensor(10000.0)) / self._positional_dim)
                    # )
                    # sinusoid = position * inv_freq
                    # pe = torch.empty(self._sequence_length - 1, self._positional_dim, device=self.device)
                    # pe[:, 0::2] = torch.sin(sinusoid)
                    # pe[:, 1::2] = torch.cos(sinusoid)
                    # self._positional_embeddings = nn.Parameter(pe.flatten(), requires_grad=False)
                    positional_embeddings_data = torch.empty(
                        (self._sequence_length - 1) * 2 * (1 if self._unified_positional_embeddings else self._n_detectors) * self._positional_dim,
                        dtype=torch.float32,
                        device=self.device
                    )
                    # Initialize with random floats in [-1, 1]
                    positional_embeddings_data.uniform_(-1.0, 1.0)
                    self._positional_embeddings = nn.Parameter(positional_embeddings_data)
                else:
                    positional_embeddings_data = torch.empty(
                        (self._sequence_length - 1) * (1 if self._unified_positional_embeddings else self._n_detectors) * self._positional_dim,
                        dtype=torch.float32,
                        device=self.device
                    )
                    # Initialize with random floats in [-1, 1]
                    positional_embeddings_data.uniform_(-1.0, 1.0)
                    self._positional_embeddings = nn.Parameter(positional_embeddings_data)

                    # if self._unified_positional_embeddings:
                    #     max_i = self._sequence_length - 1
                    #     if max_i >= (1 << self._positional_dim):
                    #         raise ValueError(
                    #             f"positional_dim={self._positional_dim} too small to represent up to i={max_i} (need >= {max_i.bit_length()} bits)")
                    #
                    #     i = torch.arange(self._sequence_length - 1, device=self.device, dtype=torch.long).unsqueeze(1)  # [L,1]
                    #     b = torch.arange(self._positional_dim, device=self.device, dtype=torch.long).unsqueeze(0)  # [1,D]
                    #     pe = ((i >> b) & 1).to(torch.float32)
                    #     self.register_buffer("_positional_embeddings_mask", pe.flatten())
            else:
                assert not self._use_sinusoidal_pe
                self._positional_embeddings = None
        else:
            assert self._positional_dim is None, "positional_dim must be None when sequence_length == 1"
            assert not self._use_sinusoidal_pe
            self._positional_embeddings = None

    def __init__(
        self, n_inputs, n_outputs,
        n_detectors, n_anchors_per_detector,
        is_fully_connected,
        sequence_length,
        synapse_metas: List[SynapseMeta],
        concatenation_product=True,
        sliced_product_mode=False,
        positional_dim=None,
        use_sinusoidal_pe=False,
        weights_gradient_policy: GradientPolicy = None,
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _initial_synapse_capacity=None,
        _forward_group_size: int = 32,
        _backward_group_size: int = 32
    ):
        super().__init__()

        assert len(synapse_metas) > 0
        assert sequence_length > 0
        self.device = torch.device("cpu")
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs

        self._n_detectors = n_detectors
        self._n_anchors_per_detector = n_anchors_per_detector
        self._is_fully_connected = is_fully_connected
        self._forward_group_size = _forward_group_size
        self._backward_group_size = _backward_group_size
        self._sequence_length = sequence_length
        self._multi_id = 0
        if self._sequence_length == 1 and not concatenation_product:
            raise ValueError("concatenation_product == False doesn't make sense with sequence_length == 1")

        self._concatenation_product = concatenation_product
        self._sliced_product_mode = sliced_product_mode

        if self._sliced_product_mode and concatenation_product:
            raise ValueError("sliced_product_mode == True doesn't make sense in concatenation product mode")

        if self._sliced_product_mode and n_inputs != positional_dim:
            raise ValueError("if sliced_product_mode == True then n_inputs and positional_dim should be equal")

        if self._concatenation_product:
            self._unified_positional_embeddings = False
        else:
            self._unified_positional_embeddings = True

        if shared_context is None:
            self._own_shared_context = True
            shared_context = LUTSharedContext()
            shared_context.to_device(self.device)
        else:
            self._own_shared_context = False

        self._shared_context = shared_context
        if weights_gradient_policy is None:
            weights_gradient_policy = GradientPolicy(GradientType.Dense, normalized=False)

        if summation_dtype == torch.int32 and weights_gradient_policy.type == GradientType.Internal:
            raise ValueError("summation dtype torch.int32 can't be combined with GradientType.Internal")

        self._weights_gradient_policy = weights_gradient_policy
        self._external_lr_hook = None

        self._positional_dim = positional_dim
        self._use_sinusoidal_pe = use_sinusoidal_pe
        self._initialize_positional_embeddings()

        if self._is_fully_connected:
            assert len(synapse_metas) == 1, "fully connected mode is not compatible with multiple synapse metas"

        if _initial_synapse_capacity is None:
            _initial_synapse_capacity = self._n_lookup_neurons * n_outputs

        if summation_dtype == torch.float32:
            self._lut_dm = LUTDataManagerF(
                n_inputs if self._concatenation_product else (2 * n_inputs + positional_dim),
                n_outputs, n_detectors, n_anchors_per_detector,
                sequence_length if self._concatenation_product else 1,
                positional_dim if positional_dim is not None else 0,
                _initial_synapse_capacity,
                _forward_group_size,
                _backward_group_size
            )
        else:
            self._lut_dm = LUTDataManagerI(
                n_inputs if self._concatenation_product else (2 * n_inputs + positional_dim),
                n_outputs, n_detectors, n_anchors_per_detector,
                sequence_length if self._concatenation_product else 1,
                positional_dim if positional_dim is not None else 0,
                _initial_synapse_capacity,
                _forward_group_size,
                _backward_group_size,
                _int_rescaler
            )

        self._n_lookup_neurons = self._lut_dm.get_number_of_lookup_neurons()
        self._synapse_metas = synapse_metas

        for i, sm in enumerate(synapse_metas):
            m_id = self._lut_dm.register_synapse_meta(
                learning_rate=sm.learning_rate,
                min_synaptic_weight=sm.min_weight,
                max_synaptic_weight=sm.max_weight,
                initial_noise_level=sm.initial_noise_level,
                initial_weight=sm.initial_weight
            )
            assert m_id == i

        self._lut_dm.initialize_neurons(is_fully_connected)
        self._input_neuron_ids = torch.arange(
            0, self._n_inputs if self._concatenation_product else 2 * self._n_inputs + self._positional_dim,
            dtype=torch.int32, device=self.device
        )
        self._detector_neuron_ids = torch.arange(
            self._input_neuron_ids.numel(), self._input_neuron_ids.numel() + self._n_detectors,
            dtype=torch.int32, device=self.device
        )
        self._lookup_neuron_ids = torch.arange(
            0, self._n_lookup_neurons, dtype=torch.int32, device=self.device
        )
        self._output_neuron_ids = torch.arange(
            self._n_lookup_neurons, self._n_lookup_neurons + n_outputs,
            dtype=torch.int32, device=self.device
        )
        self._weights = None
        self._lookup_indices_callback = None
        self._detector_anchors = None

    def add_detector_connections(
        self, chunk_of_connections: ChunkOfConnections,
        ids_shift=0,
        random_seed: int = None
    ):
        self._lut_dm.add_detector_connections(
            chunk_of_connections.get_connections(),
            chunk_of_connections.get_single_group_size(),
            ids_shift,
            random_seed
        )

    def initialize_detectors(self, compact_mode=True, seed=None):
        max_n_inputs_per_detector = self._lut_dm.finalize_detector_connections()
        assert max_n_inputs_per_detector * (max_n_inputs_per_detector - 1) >= self._n_anchors_per_detector

        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None

        self._detector_anchors = torch.zeros(
            self._n_detectors * self._n_anchors_per_detector * 2,
            dtype=torch.int32,
            device=self.device
        )

        if compact_mode:
            encoded_pairs_permutations = torch.randint(
                max_n_inputs_per_detector * (max_n_inputs_per_detector - 1),
                [self._n_detectors, max_n_inputs_per_detector],
                dtype=torch.int32, device=self.device, generator=g
            )
        else:
            noise = torch.rand(
                self._n_detectors, max_n_inputs_per_detector * (max_n_inputs_per_detector - 1),
                device=self.device, generator=g
            )
            encoded_pairs_permutations = noise.argsort(dim=1, stable=True).to(dtype=torch.int32)

        self._lut_dm.initialize_detectors(
            encoded_pairs_permutations.flatten().contiguous(),
            max_n_inputs_per_detector,
            self._detector_anchors,
            compact_mode
        )

    def get_input_neuron_ids(self):
        return self._input_neuron_ids

    def get_detector_neuron_ids(self):
        return self._detector_neuron_ids

    def get_lookup_neuron_ids(self):
        return self._lookup_neuron_ids

    def get_output_neuron_ids(self):
        return self._output_neuron_ids

    def n_inputs(self):
        return self._n_inputs

    def n_lookup_neurons(self):
        return self._n_lookup_neurons

    def n_outputs(self):
        return self._n_outputs

    def n_detectors(self):
        return self._n_detectors

    def n_anchors_per_detector(self):
        return self._n_anchors_per_detector

    def sequence_length(self):
        return self._sequence_length

    def n_synapses(self):
        if self._is_fully_connected:
            return self._n_lookup_neurons * self._n_outputs
        else:
            return self._lut_dm.get_number_of_synapses()

    def input_shape(self):
        return (self._n_outputs,)

    def output_shape(self):
        return (self._n_outputs,)

    def __repr__(self):
        return f'LUTLayerBasic({self.n_inputs()} inputs, {self.n_detectors()} detectors, {self.n_anchors_per_detector()} anchors per detector, {self.n_outputs()} outputs, {self.n_synapses()} synapses, {self._lut_dm})'

    def add_lookup_connections(
        self, chunk_of_connections: ChunkOfConnections,
        ids_shift=0,
        random_seed: int = None
    ):
        if self._is_fully_connected:
            raise RuntimeError(f"Can't add lookup connections in fully connected mode")
        self._lut_dm.add_lookup_connections(
            chunk_of_connections.get_connections(),
            chunk_of_connections.get_single_group_size(),
            ids_shift,
            random_seed
        )

    def set_external_learning_rate_hook(self, hook_fn):
        """
        Set a function that will be called during backward pass to get the external learning rate.
        This hook is only allowed with Internal gradient policy and is required when using Internal policy.

        Args:
            hook_fn: A callable that takes weights tensor as parameter and returns the learning rate (float).
                    Must not be None. Signature: hook_fn(weights: torch.Tensor) -> float

        Raises:
            ValueError: If gradient policy is not Internal or if hook_fn is None.
        """
        if self._weights_gradient_policy.type != GradientType.Internal:
            raise ValueError("external_learning_rate_hook can only be used with GradientPolicy.Internal")
        if hook_fn is None:
            raise ValueError("external_learning_rate_hook cannot be None when using GradientPolicy.Internal")
        self._external_lr_hook = hook_fn

    def _reset_shared_context(self, new_context):
        self._shared_context = new_context
        self._own_shared_context = False

    def compile_lut(self, shuffle_synapses_random_seed: int = None, _only_trainable_backwards=True):
        n_weights = self._lut_dm.get_weights_dimension()
        with torch.no_grad():
            if self._is_fully_connected:
                sm = self._synapse_metas[0]
                w = torch.rand([n_weights], dtype=torch.float32, device=self.device)
                w *= sm.initial_noise_level
                w += sm.initial_weight
                w.clip_(sm.min_weight, sm.max_weight)
            else:
                w = torch.zeros([n_weights], dtype=torch.float32, device=self.device)
                self._lut_dm.compile(_only_trainable_backwards, w, shuffle_synapses_random_seed)
        self._weights = nn.Parameter(w)
        self._lut_dm.to_device(-1)
        if self.device.type == 'cuda':
            self._lut_dm.to_device(self.device.index)

    def get_smallest_distinguishable_fraction(self) -> float:
        return self._lut_dm.get_smallest_distinguishable_fraction()

    def get_epsilon(self) -> float:
        return self._lut_dm.get_epsilon()

    def get_summation_type(self) -> torch.dtype:
        t = self._lut_dm.get_summations_data_type()
        if t == 'int32':
            return torch.int32
        elif t == 'float32':
            return torch.float32
        else:
            raise RuntimeError('Unsupported summations type: ' + t)

    def get_memory_stats(self) -> str:
        return self._lut_dm.get_memory_stats()

    def get_profiling_stats(self) -> str:
        return self._lut_dm.get_profiling_stats()

    def reset_profiler(self):
        self._lut_dm.reset_profiler()

    def _synchronize(self):
        """
        Synchronize CUDA streams for this LUT layer.
        Only works when the LUT is on CUDA. Synchronizes the streams
        corresponding to this layer's multi_id from the shared context.
        """
        if self.device.type == 'cuda':
            for stream_index in range(3):
                stream = self._shared_context.get_cuda_stream(self.device, self._multi_id, stream_index)
                if stream is not None:
                    stream.synchronize()

    def _gradient_densify_buffer_size(self, batch_size):
        """
        Calculate the densify buffer size needed for gradient processing.
        
        Args:
            batch_size: The batch size for the current operation
            
        Returns:
            Size of the densify buffer needed for sparse gradient conversion
        """
        n_weights_per_neuron = self._n_outputs if self._is_fully_connected else self._lut_dm.get_max_forward_groups_per_neuron() * self._forward_group_size
        if self._sequence_length == 1:
            return self._n_detectors * n_weights_per_neuron * batch_size
        else:
            return self._n_detectors * n_weights_per_neuron * (self._sequence_length * (self._sequence_length - 1) // 2) * batch_size

    def _process_gradients(self, target_w_grad, batch_size):
        """
        Process weight gradients according to the gradient policy.
        
        Args:
            target_w_grad: The dense gradient tensor to process
            batch_size: The batch size for the current operation
            
        Returns:
            Processed gradient tensor (sparse or dense, or None if sparse conversion resulted in empty)
        """
        if self._weights_gradient_policy.type == GradientType.Sparse:
            densify_buffer_size = self._gradient_densify_buffer_size(batch_size)
            converter = self._shared_context.get_dense_to_sparse_converter(self._multi_id)
            stream = self._shared_context.get_cuda_stream(self.device, self._multi_id, stream_index=0)
            indices, values = converter.dense_to_sparse_32(
                target_w_grad, erase_input=True,
                densify_buffers=self._shared_context.get_densify_buffers(
                    densify_buffer_size,
                    self.device,
                    self._multi_id
                ),
                stream=stream,
                decouple=True
            )
            if indices is not None:
                if self._weights_gradient_policy.normalized:
                    values /= values.abs().max().clip(1e-16)
                target_w_grad = torch.sparse_coo_tensor(
                    indices=indices.unsqueeze(0),
                    values=values,
                    size=self._weights.shape,
                    device=self.device,
                    check_invariants=False,
                    is_coalesced=True,
                    requires_grad=False
                )
            else:
                target_w_grad = None
        elif self._weights_gradient_policy.type == GradientType.Dense and self._weights_gradient_policy.normalized:
            target_w_grad /= target_w_grad.abs().max().clip(1e-16)
        
        return target_w_grad

    @staticmethod
    def _process_multiple_sparse_gradients(
        shared_context, target_w_grad_list, multi_id_list, densify_buffer_size, do_normalize
    ):
        densify_buffers_list = [None] * len(target_w_grad_list)
        results = [None] * len(target_w_grad_list)

        for i, target_w_grad in enumerate(target_w_grad_list):
            if target_w_grad is None:
                continue
                
            multi_id = multi_id_list[i]
            converter = shared_context.get_dense_to_sparse_converter(multi_id)
            densify_buffers = shared_context.get_densify_buffers(
                densify_buffer_size,
                target_w_grad.device,
                multi_id
            )
            stream = shared_context.get_cuda_stream(target_w_grad.device, multi_id, stream_index=0)
            converter.dense_to_sparse_32(
                target_w_grad, erase_input=True,
                densify_buffers=densify_buffers,
                stream=stream,
                decouple=False
            )
            densify_buffers_list[i] = densify_buffers
    
        for i, densify_buffers in enumerate(densify_buffers_list):
            if densify_buffers is None:
                continue
            target_w_grad = target_w_grad_list[i]
            converter = shared_context.get_dense_to_sparse_converter(multi_id_list[i])
            stream = shared_context.get_cuda_stream(target_w_grad.device, multi_id, stream_index=0)
            if stream is not None:
                stream.synchronize()
            indices, values = converter.decouple_results(densify_buffers)
            
            if indices is not None:
                if do_normalize:
                    values /= values.abs().max().clip(1e-16)
                target_w_grad = torch.sparse_coo_tensor(
                    indices=indices.unsqueeze(0),
                    values=values,
                    size=target_w_grad.shape,
                    device=target_w_grad.device,
                    check_invariants=False,
                    is_coalesced=True,
                    requires_grad=False
                )
            else:
                target_w_grad = None
            results[i] = target_w_grad
        return results

    def _set_lookup_indices_callback(self, cb):
        assert self._sequence_length == 1
        self._lookup_indices_callback = cb

    def forward_step(self, x, output=None):
        if not (len(x.shape) == len(self.input_shape()) + 2 and x.shape[2:] == self.input_shape()):
            raise ValueError(
                f"Input x has invalid shape {x.shape}; expected {(x.shape[0], 1, *self.input_shape())}"
                f" input_shape={self.input_shape()}"
            )

        assert self._sequence_length == 1
        assert x.device == self.device
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == 1, f"Input sequence_length {sequence_length} does not match constructor sequence_length 1"

        x = x.view(-1)
        if output is None:
            external_output = False
            output = torch.zeros([batch_size * self._n_outputs], dtype=torch.float32, device=self.device)
        else:
            external_output = True
        lookup_indices = torch.zeros(
            [batch_size * self._n_detectors], dtype=torch.int32, device=self.device
        )
        if self.training:
            min_anchor_deltas = torch.zeros(
                [batch_size * self._n_detectors], dtype=torch.float32, device=self.device, requires_grad=False
            )
            min_anchor_delta_indices = torch.zeros(
                [batch_size * self._n_detectors], dtype=torch.int32, device=self.device
            )
        else:
            min_anchor_deltas = None
            min_anchor_delta_indices = None

        stream_handles = self._shared_context.get_cuda_streams(self.device, self._multi_id) if self.device.type == 'cuda' else None

        self._lut_dm.forward_step(
            self._weights,
            batch_size, x,
            self._detector_anchors,
            output,
            lookup_indices,
            stream_handles,
            min_anchor_deltas,
            min_anchor_delta_indices
        )

        if not external_output:
            self._synchronize()
            if self.training and self._lookup_indices_callback is not None:
                self._lookup_indices_callback(lookup_indices, min_anchor_deltas, min_anchor_delta_indices)

        if self.training:
            result = () if external_output else (output.view((batch_size, 1) + self.output_shape()),)
            return result + (
                lookup_indices.view(batch_size, 1, self._n_detectors),
                min_anchor_deltas.view(batch_size, 1, self._n_detectors),
                min_anchor_delta_indices.view(batch_size, 1, self._n_detectors)
            )
        else:
            return None if external_output else output.view((batch_size, 1) + self.output_shape())

    def forward_step_concat(self, x, pos_embeddings, output=None):
        if not (len(x.shape) == len(self.input_shape()) + 2 and x.shape[2:] == self.input_shape()):
            raise ValueError(
                f"Input x has invalid shape {x.shape}; expected {(x.shape[0], 'S', *self.input_shape())} or {(x.shape[0], *self.input_shape())}"
                f"where input_shape={self.input_shape()}"
            )

        assert self._concatenation_product
        assert x.device == self.device
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == self._sequence_length, f"Input sequence_length {sequence_length} does not match constructor sequence_length {self._sequence_length}"
        expected_shape = (batch_size, sequence_length) + self.input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.view(-1)
        if output is None:
            external_output = False
            output = torch.zeros(
                [batch_size * sequence_length * self._n_outputs],
                dtype=torch.float32, device=self.device
            )
        else:
            external_output = True
        lookup_indices = torch.zeros(
            [batch_size * sequence_length * self._n_detectors], dtype=torch.int32, device=self.device
        )
        if self.training:
            min_anchor_deltas = torch.zeros(
                [batch_size * sequence_length * self._n_detectors], dtype=torch.float32,
                device=self.device, requires_grad=False
            )
            min_anchor_delta_indices = torch.zeros(
                [batch_size * sequence_length * self._n_detectors], dtype=torch.int32, device=self.device
            )
        else:
            min_anchor_deltas = None
            min_anchor_delta_indices = None
        if self._positional_dim > 0:
            positional_lookup_indices = torch.zeros(
                [(sequence_length - 1) * self._n_detectors], dtype=torch.int32, device=self.device
            )
            if self.training:
                positional_min_deltas = torch.zeros(
                    [(sequence_length - 1) * self._n_detectors], dtype=torch.float32,
                    device=self.device, requires_grad=False
                )
                positional_min_delta_indices = torch.zeros(
                    [(sequence_length - 1) * self._n_detectors], dtype=torch.int32, device=self.device
                )
            else:
                positional_min_deltas = None
                positional_min_delta_indices = None
        else:
            positional_lookup_indices = None
            positional_min_deltas = None
            positional_min_delta_indices = None

        stream_handles = self._shared_context.get_cuda_streams(self.device, self._multi_id) if self.device.type == 'cuda' else None

        self._lut_dm.forward_step_concat(
            self._weights,
            batch_size, x,
            self._detector_anchors,
            output,
            lookup_indices,
            pos_embeddings,
            positional_lookup_indices,
            min_anchor_deltas,
            min_anchor_delta_indices,
            positional_min_deltas,
            positional_min_delta_indices,
            stream_handles
        )

        if not external_output:
            self._synchronize()

        result = () if external_output else (output.view((batch_size, sequence_length) + self.output_shape()),)
        if self.training:
            if self._positional_dim > 0:
                return result + (
                    lookup_indices.view(batch_size, sequence_length, self._n_detectors),
                    min_anchor_deltas.view(batch_size, sequence_length, self._n_detectors),
                    min_anchor_delta_indices.view(batch_size, sequence_length, self._n_detectors),
                    positional_lookup_indices.view(sequence_length - 1, self._n_detectors),
                    positional_min_deltas.view(sequence_length - 1, self._n_detectors),
                    positional_min_delta_indices.view(sequence_length - 1, self._n_detectors)
                )
            else:
                return result + (
                    lookup_indices.view(batch_size, sequence_length, self._n_detectors),
                    min_anchor_deltas.view(batch_size, sequence_length, self._n_detectors),
                    min_anchor_delta_indices.view(batch_size, sequence_length, self._n_detectors),
                    None,  # positional_lookup_indices
                    None,  # positional_min_deltas
                    None   # positional_min_delta_indices
                )
        else:
            return None if external_output else output.view((batch_size, sequence_length) + self.output_shape())

    def forward_step_product(self, x, pos_embeddings, output=None):
        if not (len(x.shape) == len(self.input_shape()) + 2 and x.shape[2:] == self.input_shape()):
            raise ValueError(
                f"Input x has invalid shape {x.shape}; expected {(x.shape[0], 'S', *self.input_shape())} or {(x.shape[0], *self.input_shape())}"
                f"where input_shape={self.input_shape()}"
            )

        assert not self._concatenation_product
        assert x.device == self.device
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == self._sequence_length, f"Input sequence_length {sequence_length} does not match constructor sequence_length {self._sequence_length}"
        expected_shape = (batch_size, sequence_length) + self.input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.view(-1)
        if output is None:
            external_output = False
            output = torch.zeros(
                [batch_size * sequence_length * self._n_outputs],
                dtype=torch.float32, device=self.device
            )
        else:
            external_output = True

        stream_handles = self._shared_context.get_cuda_streams(self.device, self._multi_id) if self.device.type == 'cuda' else None

        self._lut_dm.forward_step_product(
            self._weights,
            batch_size, self._sequence_length,
            x, x,
            self._detector_anchors,
            output,
            self._n_inputs, self._n_inputs,
            self._sliced_product_mode,
            pos_embeddings,
            stream_handles
        )

        if not external_output:
            self._synchronize()

        return None if external_output else output.view((batch_size, sequence_length) + self.output_shape())

    def backward_step(
        self, x, grad_output,
        lookup_indices, min_anchor_deltas, min_anchor_delta_indices,
        x_grad=None
    ):
        assert self._sequence_length == 1
        assert x.device == self.device
        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == 1, f"Input sequence_length {sequence_length} does not match constructor sequence_length 1"
        expected_shape = (batch_size, 1) + self.input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.view(-1)
        assert lookup_indices.device == self.device
        assert lookup_indices.shape == (batch_size, 1, self._n_detectors)
        lookup_indices = lookup_indices.view(-1)
        assert min_anchor_deltas.device == self.device
        assert min_anchor_deltas.shape == (batch_size, 1, self._n_detectors)
        min_anchor_deltas = min_anchor_deltas.view(-1)
        assert min_anchor_delta_indices.device == self.device
        assert min_anchor_delta_indices.shape == (batch_size, 1, self._n_detectors)
        min_anchor_delta_indices = min_anchor_delta_indices.view(-1)

        if x_grad is None:
            external_output = False
            x_grad = torch.zeros_like(x)
        else:
            external_output = True
        if self._weights_gradient_policy.type == GradientType.Internal:
            target_w_grad = None
        elif self._weights_gradient_policy.type == GradientType.Sparse:
            numel = ((self._weights.numel() + 3) // 4) * 4
            target_w_grad = self._shared_context.get_weight_gradients_buffer(
                numel,
                self.device,
                self._multi_id
            )
        else:
            target_w_grad = torch.zeros_like(self._weights, requires_grad=False)

        assert grad_output.device == self.device
        assert grad_output.shape == (batch_size, 1) + self.output_shape()

        grad_output = grad_output.contiguous().view(-1)

        if self._weights_gradient_policy.type == GradientType.Internal:
            if self._external_lr_hook is None:
                raise ValueError("external_learning_rate_hook must be set when using GradientPolicy.Internal")
            external_lr = self._external_lr_hook(self._weights)
        else:
            external_lr = -1.0

        stream_handles = self._shared_context.get_cuda_streams(self.device, self._multi_id) if self.device.type == 'cuda' else None

        self._lut_dm.backward_backprop(
            self._weights,
            batch_size,
            grad_output,
            self._detector_anchors,
            lookup_indices,
            min_anchor_deltas,
            min_anchor_delta_indices,
            x_grad,
            external_lr,
            target_w_grad if self._weights_gradient_policy.type != GradientType.Internal else None,
            stream_handles
        )

        if not external_output:
            self._synchronize()
            target_w_grad = self._process_gradients(target_w_grad, batch_size)

        result = () if external_output else (x_grad.view(source_x_shape),)
        return result + (target_w_grad,)

    def backward_step_concat(
        self, x, grad_output,
        lookup_indices, min_anchor_deltas, min_anchor_delta_indices,
        positional_lookup_indices, positional_min_deltas,
        positional_min_delta_indices, x_grad=None
    ):
        assert x.device == self.device
        assert self._concatenation_product
        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == self._sequence_length, f"Input sequence_length {sequence_length} does not match constructor sequence_length {self._sequence_length}"
        expected_shape = (batch_size, sequence_length) + self.input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.view(-1)
        assert lookup_indices.device == self.device
        assert lookup_indices.shape == (batch_size, sequence_length, self._n_detectors)
        lookup_indices = lookup_indices.view(-1)
        assert min_anchor_deltas.device == self.device
        assert min_anchor_deltas.shape == (batch_size, sequence_length, self._n_detectors)
        min_anchor_deltas = min_anchor_deltas.view(-1)
        assert min_anchor_delta_indices.device == self.device
        assert min_anchor_delta_indices.shape == (batch_size, sequence_length, self._n_detectors)
        min_anchor_delta_indices = min_anchor_delta_indices.view(-1)

        if self._positional_dim > 0:
            assert positional_lookup_indices.device == self.device
            assert positional_lookup_indices.shape == (sequence_length - 1, self._n_detectors)
            positional_lookup_indices = positional_lookup_indices.view(-1)
            assert positional_min_deltas.device == self.device
            assert positional_min_deltas.shape == (sequence_length - 1, self._n_detectors)
            positional_min_deltas = positional_min_deltas.view(-1)
            assert positional_min_delta_indices.device == self.device
            assert positional_min_delta_indices.shape == (sequence_length - 1, self._n_detectors)
            positional_min_delta_indices = positional_min_delta_indices.view(-1)
        else:
            assert positional_lookup_indices is None
            assert positional_min_deltas is None
            assert positional_min_delta_indices is None
            positional_lookup_indices = None
            positional_min_deltas = None
            positional_min_delta_indices = None

        if x_grad is None:
            external_output = False
            x_grad = torch.zeros_like(x)
        else:
            external_output = True
        if self._weights_gradient_policy.type == GradientType.Internal:
            target_w_grad = None
        elif self._weights_gradient_policy.type == GradientType.Sparse:
            numel = ((self._weights.numel() + 3) // 4) * 4
            target_w_grad = self._shared_context.get_weight_gradients_buffer(
                numel,
                self.device,
                self._multi_id
            )
        else:
            target_w_grad = torch.zeros_like(self._weights, requires_grad=False)
        if self._positional_dim > 0:
            positional_grad = torch.zeros(
                [(self._sequence_length - 1) * self._n_detectors * self._positional_dim],
                requires_grad=False, device=self.device
            )
        else:
            positional_grad = None

        assert grad_output.device == self.device
        assert grad_output.shape == (batch_size, sequence_length) + self.output_shape()

        grad_output = grad_output.view(-1)

        if self._weights_gradient_policy.type == GradientType.Internal:
            if self._external_lr_hook is None:
                raise ValueError("external_learning_rate_hook must be set when using GradientPolicy.Internal")
            external_lr = self._external_lr_hook(self._weights)
        else:
            external_lr = -1.0

        stream_handles = self._shared_context.get_cuda_streams(self.device, self._multi_id) if self.device.type == 'cuda' else None

        self._lut_dm.backward_backprop_concat(
            self._weights,
            batch_size,
            grad_output,
            self._detector_anchors,
            lookup_indices,
            min_anchor_deltas,
            min_anchor_delta_indices,
            x_grad,
            external_lr,
            positional_lookup_indices,
            positional_min_deltas,
            positional_min_delta_indices,
            positional_grad,
            target_w_grad if self._weights_gradient_policy.type != GradientType.Internal else None,
            stream_handles
        )

        if not external_output:
            self._synchronize()
            target_w_grad = self._process_gradients(target_w_grad, batch_size)

        if self._positional_dim > 0 and self._unified_positional_embeddings:
            positional_grad = positional_grad.reshape(
                self._sequence_length - 1, self._n_detectors, self._positional_dim
            ).sum(dim=1)

        if self._positional_dim > 0 and self._weights_gradient_policy.normalized:
            positional_grad /= positional_grad.abs().max().clip(1e-16)

        result = () if external_output else (x_grad.view(source_x_shape),)
        return result + (target_w_grad, positional_grad,)

    def backward_step_product(
        self, x, pos_embeddings, grad_output, x_grad=None
    ):
        assert x.device == self.device
        assert not self._concatenation_product

        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == self._sequence_length, f"Input sequence_length {sequence_length} does not match constructor sequence_length {self._sequence_length}"
        expected_shape = (batch_size, sequence_length) + self.input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.view(-1)

        if x_grad is None:
            external_output = False
            x_grad = torch.zeros_like(x)
        else:
            external_output = True
        if self._weights_gradient_policy.type == GradientType.Internal:
            target_w_grad = None
        elif self._weights_gradient_policy.type == GradientType.Sparse:
            numel = ((self._weights.numel() + 3) // 4) * 4
            target_w_grad = self._shared_context.get_weight_gradients_buffer(
                numel,
                self.device,
                self._multi_id
            )
        else:
            target_w_grad = torch.zeros_like(self._weights, requires_grad=False)
        if self._positional_dim > 0:
            positional_grad = torch.zeros_like(pos_embeddings, requires_grad=False)
        else:
            positional_grad = None

        assert grad_output.device == self.device
        assert grad_output.shape == (batch_size, sequence_length) + self.output_shape()

        grad_output = grad_output.view(-1)

        if self._weights_gradient_policy.type == GradientType.Internal:
            if self._external_lr_hook is None:
                raise ValueError("external_learning_rate_hook must be set when using GradientPolicy.Internal")
            external_lr = self._external_lr_hook(self._weights)
        else:
            external_lr = -1.0

        stream_handles = self._shared_context.get_cuda_streams(self.device, self._multi_id) if self.device.type == 'cuda' else None

        self._lut_dm.backward_backprop_product(
            self._weights,
            batch_size,
            self._sequence_length,
            x, x,
            self._detector_anchors,
            grad_output,
            x_grad, x_grad,
            external_lr,
            self._n_inputs, self._n_inputs,
            self._sliced_product_mode,
            target_w_grad if self._weights_gradient_policy.type != GradientType.Internal else None,
            pos_embeddings,
            positional_grad,
            stream_handles
        )

        if not external_output:
            self._synchronize()
            target_w_grad = self._process_gradients(target_w_grad, batch_size)

        if self._positional_dim > 0 and self._weights_gradient_policy.normalized:
            positional_grad /= positional_grad.abs().max().clip(1e-16)

        result = () if external_output else (x_grad.view(source_x_shape),)
        return result + (target_w_grad, positional_grad,)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]

        dev = torch.device(device)
        if dev.type == 'cuda' and dev.index is None:
            device_index = torch.cuda.current_device()
            dev = torch.device(f'cuda:{device_index}')
        elif dev.index is not None:
            device_index = dev.index
        else:
            device_index = -1

        self.device = dev
        self._input_neuron_ids = self._input_neuron_ids.to(device=self.device)
        self._detector_neuron_ids = self._detector_neuron_ids.to(device=self.device)
        self._lookup_neuron_ids = self._lookup_neuron_ids.to(device=self.device)
        self._output_neuron_ids = self._output_neuron_ids.to(device=self.device)

        self._lut_dm.to_device(device_index)
        if self._detector_anchors is not None:
            self._detector_anchors = self._detector_anchors.to(device=self.device)
        if self._own_shared_context:
            self._shared_context.to_device(self.device)
        return self

    class LUTForwardFN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            x, _, pos_embeddings, lut_layer = args
            if lut_layer.training:
                ctx.lut_layer = lut_layer
                if lut_layer._sequence_length == 1:
                    (
                        output, lookup_indices, min_anchor_deltas,
                        min_anchor_delta_indices
                    ) = lut_layer.forward_step(x)
                    ctx.save_for_backward(x, lookup_indices, min_anchor_deltas, min_anchor_delta_indices)
                elif lut_layer._concatenation_product:
                    (
                        output, lookup_indices, min_anchor_deltas,
                        min_anchor_delta_indices, positional_lookup_indices,
                        positional_min_deltas, positional_min_delta_indices
                    ) = lut_layer.forward_step_concat(x, pos_embeddings)
                    ctx.save_for_backward(
                        x, lookup_indices, min_anchor_deltas, min_anchor_delta_indices,
                        positional_lookup_indices, positional_min_deltas,
                        positional_min_delta_indices
                    )
                else:
                    output = lut_layer.forward_step_product(x, pos_embeddings)
                    ctx.save_for_backward(x, pos_embeddings)
                return output
            elif lut_layer._sequence_length == 1:
                return lut_layer.forward_step(x)
            elif lut_layer._concatenation_product:
                return lut_layer.forward_step_concat(x, pos_embeddings)
            else:
                return lut_layer.forward_step_product(x, pos_embeddings)

        @staticmethod
        def backward(ctx, *grad_outputs):
            (grad_output,) = grad_outputs
            if ctx.lut_layer._sequence_length == 1:
                (x, lookup_indices, min_anchor_deltas, min_anchor_delta_indices) = ctx.saved_tensors
                x_grad, w_grad = ctx.lut_layer.backward_step(
                    x, grad_output,
                    lookup_indices,
                    min_anchor_deltas,
                    min_anchor_delta_indices
                )
                return x_grad, w_grad, None, None
            elif ctx.lut_layer._concatenation_product:
                (
                    x, lookup_indices, min_anchor_deltas, min_anchor_delta_indices, positional_lookup_indices,
                    positional_min_deltas, positional_min_delta_indices
                ) = ctx.saved_tensors
                x_grad, w_grad, pe_grad = ctx.lut_layer.backward_step_concat(
                    x, grad_output,
                    lookup_indices,
                    min_anchor_deltas,
                    min_anchor_delta_indices,
                    positional_lookup_indices,
                    positional_min_deltas,
                    positional_min_delta_indices
                )
                return x_grad, w_grad, pe_grad, None
            else:
                (x, pos_embeddings) = ctx.saved_tensors
                x_grad, w_grad, pe_grad = ctx.lut_layer.backward_step_product(x, pos_embeddings, grad_output)
                return x_grad, w_grad, pe_grad, None

    def forward(self, x):
        if self._sequence_length == 1:
            return LUTLayerBasic.LUTForwardFN.apply(x, self._weights, None, self)
        else:
            if self._unified_positional_embeddings and self._concatenation_product:
                pos_embeddings = self._positional_embeddings.reshape(
                    (self._sequence_length - 1) * (2 if self._use_sinusoidal_pe else 1), 1, self._positional_dim
                ).repeat(1, self._n_detectors, 1).flatten().contiguous()
            else:
                pos_embeddings = self._positional_embeddings

            if self._use_sinusoidal_pe:
                position = torch.arange(self._sequence_length - 1, device=self.device).to(dtype=torch.float32)
                position = position.unsqueeze(1).repeat(1, self._positional_dim * (self._n_detectors if self._concatenation_product else 1))
                pos_embeddings = pos_embeddings.reshape(pos_embeddings.numel() // 2, 2)
                pos_embeddings = torch.sin(position.flatten() * pos_embeddings[:, 0] + pos_embeddings[:, 1])
                pos_embeddings = pos_embeddings.flatten().contiguous()

            return LUTLayerBasic.LUTForwardFN.apply(x, self._weights, pos_embeddings, self)

    def _count_synapses(self, neuron_ids: torch.Tensor):
        if self._is_fully_connected:
            return self._n_lookup_neurons * self._n_outputs
        else:
            return self._lut_dm.count_synapses(neuron_ids)

    def _export_synapses(
        self, neuron_ids: torch.Tensor,
        source_ids: torch.Tensor,
        weights: torch.Tensor,
        target_ids: torch.Tensor,
        synapse_metas: torch.Tensor = None
    ):
        if self._is_fully_connected:
            source_ids[:] = neuron_ids.view(neuron_ids.numel(), -1).repeat(1, self._n_outputs).flatten()
            weights[:] = self._weights.view(self._n_lookup_neurons, self._n_outputs)[neuron_ids].flatten()
            target_ids[:] = torch.arange(
                self._n_outputs, device=self.device, dtype=torch.int32
            ).repeat(neuron_ids.numel())
            target_ids += self._n_lookup_neurons
            if synapse_metas is not None:
                synapse_metas[:] = 0.0
        else:
            self._lut_dm.export_synapses(
                self._weights,
                neuron_ids,
                source_ids,
                weights,
                target_ids,
                synapse_metas
            )

    def _export_anchors(self):
        """
        Export all anchor pairs for all detectors.

        Returns:
            torch.Tensor: A tensor of shape [n_detectors, n_anchors_per_detector, 2]
                         containing anchor pairs. Each anchor pair is [anchor1_id, anchor2_id].
        """
        if self._detector_anchors is None:
            raise RuntimeError("Detectors are not initialized, call initialize_detectors() first")
        return self._detector_anchors.view(self._n_detectors, self._n_anchors_per_detector, 2)


class Conv2DLUTLayer(LUTLayerBasic):
    def __init__(
        self, input_shape,
        n_anchors_per_detector,
        detectors_shape,
        output_kernel_shape,
        sequence_length=1,
        receptive_field_shape=None,
        receptive_field_stride_shape=None,
        lut_receptive_field_shape=None,
        lut_receptive_field_stride_shape=None,
        synapse_meta=SynapseMeta(),
        concatenation_product=True,
        sliced_product_mode=False,
        positional_dim=None,
        use_sinusoidal_pe=False,
        weights_gradient_policy: GradientPolicy = None,
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
        _explicit_anchors=None,
        _int_rescaler=0.001,
        _forward_group_size=32,
        _backward_group_size=32,
        _max_groups_in_growth_buffer=2 ** 20,
        random_seed=None,
        device=None
    ):
        if concatenation_product:
            input_shape_ex = input_shape
        elif sliced_product_mode:
            input_shape_ex = (input_shape[0], input_shape[1] * 3)
        else:
            input_shape_ex = (input_shape[0], input_shape[1] * 2 + (0 if positional_dim is None else positional_dim))

        if receptive_field_shape is None:
            assert receptive_field_stride_shape is None
            receptive_field_shape_ex = input_shape_ex
            receptive_field_stride_shape_ex = input_shape_ex
        else:
            if concatenation_product:
                receptive_field_shape_ex = receptive_field_shape
                receptive_field_stride_shape_ex = receptive_field_stride_shape
            elif sliced_product_mode:
                receptive_field_shape_ex = (receptive_field_shape[0], receptive_field_shape[1] * 3)
                receptive_field_stride_shape_ex = (receptive_field_stride_shape[0], receptive_field_stride_shape[1] * 3)
            else:
                assert False

        c_helper_1 = Conv2DSynapseGrowthHelper(
            input_shape_ex[0], input_shape_ex[1],
            receptive_field_shape_ex[0], receptive_field_shape_ex[1],
            receptive_field_stride_shape_ex[0], receptive_field_stride_shape_ex[1],
            detectors_shape[0], detectors_shape[1]
        )

        lut_shape = c_helper_1.out_h, c_helper_1.out_w
        n_lut_channels = LUTLayerBasic.n_lut_channels(n_anchors_per_detector, sequence_length)

        n_inputs = input_shape[0] * input_shape[1]
        n_detectors = lut_shape[0] * lut_shape[1]

        if lut_receptive_field_shape is not None:
            c_helper_2 = Conv2DSynapseGrowthHelper(
                lut_shape[0], lut_shape[1],
                lut_receptive_field_shape[0], lut_receptive_field_shape[1],
                lut_receptive_field_stride_shape[0], lut_receptive_field_stride_shape[1],
                output_kernel_shape[0], output_kernel_shape[1],
                n_input_channels=n_lut_channels
            )
            n_outputs = c_helper_2.out_h * c_helper_2.out_w
            self._lut_receptive_field_shape = lut_receptive_field_shape + (n_lut_channels,)
            self._output_shape = (c_helper_2.out_h, c_helper_2.out_w)
        else:
            assert lut_receptive_field_stride_shape is None
            c_helper_2 = None
            n_outputs = output_kernel_shape[0] * output_kernel_shape[1]
            self._lut_receptive_field_shape = lut_shape + (n_lut_channels,)
            self._output_shape = output_kernel_shape

        self._input_shape = input_shape
        self._lut_shape = lut_shape + (n_lut_channels,)
        self._detectors_shape = detectors_shape

        super().__init__(
            n_inputs=n_inputs, n_outputs=n_outputs, n_detectors=n_detectors,
            n_anchors_per_detector=n_anchors_per_detector, is_fully_connected=c_helper_2 is None,
            sequence_length=sequence_length, synapse_metas=[synapse_meta],
            concatenation_product=concatenation_product, sliced_product_mode=sliced_product_mode,
            positional_dim=positional_dim, use_sinusoidal_pe=use_sinusoidal_pe,
            weights_gradient_policy=weights_gradient_policy,
            shared_context=shared_context,
            summation_dtype=summation_dtype, _int_rescaler=_int_rescaler,
            _initial_synapse_capacity=0 if c_helper_2 is None else c_helper_2.n_connections(),
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size,
        )

        if device is not None:
            self.to(device=device)
        else:
            device = torch.device("cpu")

        if _explicit_anchors is None:
            connections = c_helper_1.grow_synapses(
                input_ids=self.get_input_neuron_ids().view(input_shape_ex) + 1,
                output_ids=self.get_detector_neuron_ids().view(lut_shape[0], lut_shape[1]) + 1,
                max_groups_in_buffer=_max_groups_in_growth_buffer,
                device=device,
                seed=random_seed
            )

            self.add_detector_connections(
                chunk_of_connections=connections,
                ids_shift=-1,
                random_seed=random_seed
            )

            self.initialize_detectors(seed=random_seed)
        else:
            self._detector_anchors = _explicit_anchors.to(self.device)

        if c_helper_2 is not None:
            connections = c_helper_2.grow_synapses(
                input_ids=self.get_lookup_neuron_ids().view(lut_shape + (n_lut_channels,)) + 1,
                output_ids=self.get_output_neuron_ids().view(c_helper_2.out_h, c_helper_2.out_w) + 1,
                device=device,
                seed=random_seed
            )

            self.add_lookup_connections(
                chunk_of_connections=connections,
                ids_shift=-1,
                random_seed=random_seed
            )

        self.compile_lut()

    def input_shape(self):
        return self._input_shape

    def output_shape(self):
        return self._output_shape

    def lut_shape(self):
        return self._lut_shape

    def detectors_shape(self):
        return self._detectors_shape

    def lut_receptive_field_shape(self):
        return self._lut_receptive_field_shape

    def __repr__(self):
        return f'Conv2DLUTLayer(input_shape={self.input_shape()}, output_shape={self.output_shape()}, detectors_shape={self.detectors_shape()}, n_anchors_per_detector={self.n_anchors_per_detector()})'

    def export_weights(self, inverse_order=True):
        n_synapses = self.n_synapses()
        source_ids = torch.zeros([n_synapses], dtype=torch.int32, device=self.device)
        target_ids = torch.zeros([n_synapses], dtype=torch.int32, device=self.device)
        weights = torch.zeros([n_synapses], dtype=torch.float32, device=self.device)

        self._export_synapses(
            self.get_lookup_neuron_ids(),
            source_ids,
            weights,
            target_ids
        )

        if inverse_order:
            order = torch.argsort(source_ids, stable=True, descending=False)
            order = order[torch.argsort(target_ids[order], stable=True, descending=False)]
            return weights[order].view(self.output_shape() + self.lut_receptive_field_shape())
        else:
            order = torch.argsort(target_ids, stable=True, descending=False)
            order = order[torch.argsort(source_ids[order], stable=True, descending=False)]
            return weights[order].view(self.lut_receptive_field_shape() + self.output_shape())


class LUTLayer(Conv2DLUTLayer):
    def __init__(
        self, n_inputs,
        n_anchors_per_detector,
        n_detectors,
        n_outputs,
        sequence_length=1,
        synapse_meta=SynapseMeta(),
        concatenation_product=True,
        sliced_product_mode=False,
        positional_dim=None,
        use_sinusoidal_pe=False,
        weights_gradient_policy: GradientPolicy = None,
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
        _explicit_anchors=None,
        _int_rescaler=0.001,
        _forward_group_size=32,
        _backward_group_size=32,
        _max_groups_in_growth_buffer=2 ** 20,
        random_seed=None,
        device=None
    ):
        super().__init__(
            input_shape=(1, n_inputs,),
            n_anchors_per_detector=n_anchors_per_detector,
            detectors_shape=(1, n_detectors,),
            output_kernel_shape=(1, n_outputs),
            sequence_length=sequence_length,
            receptive_field_shape=None,
            receptive_field_stride_shape=None,
            lut_receptive_field_shape=None,
            lut_receptive_field_stride_shape=None,
            synapse_meta=synapse_meta,
            concatenation_product=concatenation_product,
            sliced_product_mode=sliced_product_mode,
            positional_dim=positional_dim,
            use_sinusoidal_pe=use_sinusoidal_pe,
            weights_gradient_policy=weights_gradient_policy,
            shared_context=shared_context,
            summation_dtype=summation_dtype,
            _explicit_anchors=_explicit_anchors,
            _int_rescaler=_int_rescaler,
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size,
            _max_groups_in_growth_buffer=_max_groups_in_growth_buffer,
            random_seed=random_seed,
            device=device
        )

    def input_shape(self):
        return self._input_shape[1:]

    def output_shape(self):
        return self._output_shape[1:]

    def detectors_shape(self):
        return self._detectors_shape[1:]

    def lut_shape(self):
        return self._lut_shape[1:]

    def lut_receptive_field_shape(self):
        return self._lut_receptive_field_shape[1:]

    def __repr__(self):
        return f'LUTLayer({self.n_inputs()} inputs, {self.n_detectors()} detectors, {self.n_outputs()} outputs, {self.n_anchors_per_detector()} anchors per detector)'


class MultiLUT(nn.Module):
    """
    A module that runs multiple LUTLayerBasic instances in parallel (cuda kernels work in parallel on different cuda streams, cpu part is synchronous).
    All layers must have the same input_shape, output_shape, and sequence_length.
    Each layer is assigned a unique multi_id for stream isolation.
    """

    def __init__(self, luts: List[LUTLayerBasic]):
        super().__init__()

        if len(luts) < 2:
            raise ValueError("MultiLUT requires at least two luts")

        # Validate all luts have compatible shapes
        first_lut = luts[0]
        input_shape = first_lut.input_shape()
        output_shape = first_lut.output_shape()
        sequence_length = first_lut.sequence_length()
        concatenation_product = first_lut._concatenation_product
        sliced_product_mode = first_lut._sliced_product_mode
        self._shared_context = first_lut._shared_context
        self._gradient_policy = first_lut._weights_gradient_policy

        for i, lut in enumerate(luts):
            if not isinstance(lut, LUTLayerBasic):
                raise ValueError(f"lut {i} is not an instance of LUTLayerBasic")
            if lut.input_shape() != input_shape:
                raise ValueError(f"lut {i} has input_shape {lut.input_shape()}, expected {input_shape}")
            if lut.output_shape() != output_shape:
                raise ValueError(f"lut {i} has output_shape {lut.output_shape()}, expected {output_shape}")
            if lut.sequence_length() != sequence_length:
                raise ValueError(f"lut {i} has sequence_length {lut.sequence_length()}, expected {sequence_length}")
            if lut._concatenation_product != concatenation_product:
                raise ValueError(f"lut {i} has _concatenation_product {lut._concatenation_product}, expected {concatenation_product}")
            if lut._sliced_product_mode != sliced_product_mode:
                raise ValueError(f"lut {i} has _sliced_product_mode {lut._sliced_product_mode}, expected {sliced_product_mode}")
            if lut._shared_context != self._shared_context:
                raise ValueError(f"lut {i} has different shared context than lut 0")
            if lut._weights_gradient_policy != self._gradient_policy:
                raise ValueError(f"lut {i} has gradient policy {lut._weights_gradient_policy}, expected {self._gradient_policy}")
            if lut.get_summation_type() == torch.int32:
                raise ValueError(f"lut {i} has summation type int32 which is not allowed for MultiLUT")

        # Assign multi_id to each lut
        for multi_id, lut in enumerate(luts):
            lut._multi_id = multi_id

        self.luts = nn.ModuleList(luts)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._sequence_length = sequence_length
        self._all_weights = None
        self._all_positional_encodings = None

    def input_shape(self):
        return self._input_shape

    def output_shape(self):
        return self._output_shape

    def sequence_length(self):
        return self._sequence_length

    def set_external_learning_rate_hook(self, hook_fn):
        for lut in self.luts:
            lut.set_external_learning_rate_hook(hook_fn)

    def _reset_shared_context(self, new_context):
        for lut in self.luts:
            lut._reset_shared_context(new_context)

    def forward(self, x):
        """
        Forward pass that runs all luts in "parallel".
        Returns the sum of outputs from all luts.
        """
        # Lazy fill weights and positional encodings if needed
        if self._all_weights is None:
            self._all_weights = []
            for lut in self.luts:
                self._all_weights.append(lut._weights)

        # Only pass positional encodings if they exist (when sequence_length > 1)
        if self._sequence_length > 1:
            if self._all_positional_encodings is None:
                self._all_positional_encodings = []
                for lut in self.luts:
                    self._all_positional_encodings.append(lut._positional_embeddings)
            return MultiLUT.MultiLUTForwardFN.apply(x, self, *self._all_weights, *self._all_positional_encodings)
        else:
            return MultiLUT.MultiLUTForwardFN.apply(x, self, *self._all_weights)

    class MultiLUTForwardFN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            """
            Run forward pass for all luts in "parallel"
            All luts accumulate into the same output tensor.
            """
            x, multi_lut = args[0], args[1]

            first_lut = multi_lut.luts[0]
            batch_size = x.shape[0]

            # Create shared output tensor
            if multi_lut._sequence_length == 1:
                output = torch.zeros(
                    [batch_size * first_lut._n_outputs],
                    dtype=torch.float32, device=first_lut.device
                )
            else:
                output = torch.zeros(
                    [batch_size * multi_lut._sequence_length * first_lut._n_outputs],
                    dtype=torch.float32, device=first_lut.device
                )

            if multi_lut.training:
                results = [None] * len(multi_lut.luts)

                for lut_idx, lut in enumerate(multi_lut.luts):
                    if multi_lut._sequence_length == 1:
                        results[lut_idx] = lut.forward_step(x, output=output)
                    elif lut._concatenation_product:
                        results[lut_idx] = list(lut.forward_step_concat(x, output=output))
                    else:
                        lut.forward_step_product(x, output=output)

                for lut_idx, lut in enumerate(multi_lut.luts):
                    lut._synchronize()
                    if lut._lookup_indices_callback is not None:
                        lut._lookup_indices_callback(results[lut_idx][0], results[lut_idx][1], results[lut_idx][2])
            else:
                for lut_idx, lut in enumerate(multi_lut.luts):
                    if multi_lut._sequence_length == 1:
                        lut.forward_step(x, output=output)
                    elif lut._concatenation_product:
                        lut.forward_step_concat(x, output=output)
                    else:
                        lut.forward_step_product(x, output=output)

                for lut_idx, lut in enumerate(multi_lut.luts):
                    lut._synchronize()

            # Reshape output to match expected shape

            output = output.view((batch_size, multi_lut._sequence_length) + multi_lut.output_shape())

            if multi_lut.training:
                # Save for backward
                ctx.multi_lut = multi_lut
                ctx.save_for_backward(x)
                ctx.results = results  # Store individual results for backward

            return output

        @staticmethod
        def backward(ctx, *grad_outputs):
            """
            Run backward pass for all luts in "parallel".
            All luts accumulate into the same x_grad tensor.
            Returns gradients for x, multi_lut, all weights, and all positional encodings.
            """
            (grad_output,) = grad_outputs
            multi_lut = ctx.multi_lut
            n_luts = len(multi_lut.luts)

            # Extract saved input
            x = ctx.saved_tensors[0]
            results = ctx.results

            # Create shared x_grad tensor
            x_grad = torch.zeros_like(x.view(-1))

            # Store gradients for each lut
            all_weight_grads = [None] * n_luts
            all_pe_grads = [None] * n_luts if multi_lut._sequence_length > 1 else []

            for lut_idx, lut in enumerate(multi_lut.luts):
                if multi_lut._sequence_length == 1:
                    (w_grad,) = lut.backward_step(
                        x, grad_output,
                        results[lut_idx][0],  # lookup_indices
                        results[lut_idx][1],  # min_anchor_deltas
                        results[lut_idx][2],  # min_anchor_delta_indices
                        x_grad=x_grad
                    )
                    all_weight_grads[lut_idx] = w_grad
                elif lut._concatenation_product:
                    w_grad, pe_grad = lut.backward_step_concat(
                        x, grad_output,
                        results[lut_idx][0],  # lookup_indices
                        results[lut_idx][1],  # min_anchor_deltas
                        results[lut_idx][2],  # min_anchor_delta_indices
                        results[lut_idx][3],  # positional_lookup_indices
                        results[lut_idx][4],  # positional_min_deltas
                        results[lut_idx][5],  # positional_min_delta_indices
                        x_grad=x_grad
                    )
                    all_weight_grads[lut_idx] = w_grad
                    all_pe_grads[lut_idx] = pe_grad
                else:
                    w_grad, pe_grad = lut.backward_step_product(
                        x, grad_output,
                        x_grad=x_grad
                    )
                    all_weight_grads[lut_idx] = w_grad
                    all_pe_grads[lut_idx] = pe_grad

            for i, lut in enumerate(multi_lut.luts):
                lut._synchronize()

            batch_size = x.shape[0]

            if multi_lut._gradient_policy.type == GradientType.Sparse:
                densify_buffer_size = multi_lut.luts[0]._gradient_densify_buffer_size(batch_size)
                all_weight_grads = LUTLayerBasic._process_multiple_sparse_gradients(
                    multi_lut._shared_context, all_weight_grads, [lut._multi_id for lut in multi_lut.luts],
                    densify_buffer_size, multi_lut._gradient_policy.normalized
                )
            else:
                for lut_idx, lut in enumerate(multi_lut.luts):
                    all_weight_grads[lut_idx] = lut._process_gradients(all_weight_grads[lut_idx], batch_size)

            if multi_lut._sequence_length > 1:
                return (x_grad.view(x.shape), None) + tuple(all_weight_grads) + tuple(all_pe_grads)
            else:
                return (x_grad.view(x.shape), None) + tuple(all_weight_grads)

    def to(self, *args, **kwargs):
        """
        Move the module to a different device/dtype.
        Resets cached weights and positional encodings so they are rebuilt on next forward.
        """
        result = super().to(*args, **kwargs)
        self._all_weights = None
        self._all_positional_encodings = None
        return result

    def __repr__(self):
        return f'MultiLUT({len(self.luts)} luts, input_shape={self.input_shape()}, output_shape={self.output_shape()}, sequence_length={self.sequence_length()})'
