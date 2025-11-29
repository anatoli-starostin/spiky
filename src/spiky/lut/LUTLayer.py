import torch
import torch.nn as nn
from typing import List, Dict, Tuple, AnyStr
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue

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
        self._sparse_firing_buffers = []
        self._before_detectors_gradients_buffers = []
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

    def get_sparse_firing_buffer(self, numel, device, multi_id=0):
        return self._ensure_buffer(self._sparse_firing_buffers, multi_id, numel, torch.int32, device)

    def get_before_detectors_gradients_buffer(self, numel, device, multi_id=0):
        buf = self._ensure_buffer(self._before_detectors_gradients_buffers, multi_id, numel, torch.float32, device)
        if self._do_asserts:
            assert torch.count_nonzero(buf) == 0
        return buf

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

    def to_device(self, device):
        dev = device if isinstance(device, torch.device) else torch.device(device)
        self._device = dev
        for i, buf in enumerate(self._sparse_firing_buffers):
            if buf is not None:
                self._sparse_firing_buffers[i] = buf.to(device=dev)
        for i, buf in enumerate(self._before_detectors_gradients_buffers):
            if buf is not None:
                self._before_detectors_gradients_buffers[i] = buf.to(device=dev)
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
    def n_lut_channels(n_anchors_per_detector, sequence_length):
        return 1 << (n_anchors_per_detector * (2 if (sequence_length > 1) else 1))

    def __init__(
        self, n_inputs, n_outputs,
        n_detectors, n_anchors_per_detector,
        is_fully_connected,
        sequence_length,
        synapse_metas: List[SynapseMeta],
        positional_dim=None,
        weights_gradient_policy: GradientPolicy = GradientPolicy(GradientType.Dense, normalized=False),
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _initial_synapse_capacity=None,
        _forward_group_size: int = 64,
        _backward_group_size: int = 8,
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

        if shared_context is None:
            self._own_shared_context = True
            shared_context = LUTSharedContext()
            shared_context.to_device(self.device)
        else:
            self._own_shared_context = False

        self._shared_context = shared_context
        self._weights_gradient_policy = weights_gradient_policy
        self._external_lr_hook = None

        # Handle positional embeddings
        if sequence_length > 1:
            assert positional_dim is not None, "positional_dim must be provided when sequence_length > 1"
            positional_embeddings_data = torch.empty(
                sequence_length * n_detectors * positional_dim,
                dtype=torch.float32,
                device=self.device
            )
            # Initialize with random floats in [-1, 1]
            positional_embeddings_data.uniform_(-1.0, 1.0)
            self._positional_embeddings = nn.Parameter(positional_embeddings_data)
        else:
            assert positional_dim is None, "positional_dim must be None when sequence_length == 1"
            self._positional_embeddings = None

        if self._is_fully_connected:
            assert len(synapse_metas) == 1, "fully connected mode is not compatible with multiple synapse metas"

        if _initial_synapse_capacity is None:
            _initial_synapse_capacity = self._n_lookup_neurons * n_outputs

        if summation_dtype == torch.float32:
            self._lut_dm = LUTDataManagerF(
                n_inputs, n_outputs, n_detectors, n_anchors_per_detector,
                sequence_length,
                positional_dim if positional_dim is not None else 0,
                _initial_synapse_capacity,
                _forward_group_size,
                _backward_group_size
            )
        else:
            self._lut_dm = LUTDataManagerI(
                n_inputs, n_outputs, n_detectors, n_anchors_per_detector,
                sequence_length,
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
            0, self._n_inputs, dtype=torch.int32, device=self.device
        )
        self._detector_neuron_ids = torch.arange(
            self._n_inputs, self._n_inputs + self._n_detectors,
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

    def initialize_detectors(self, seed=None):
        max_n_inputs_per_detector = self._lut_dm.finalize_detector_connections()
        assert max_n_inputs_per_detector * (max_n_inputs_per_detector - 1) >= self._n_anchors_per_detector

        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None
        noise = torch.rand(
            self._n_detectors, max_n_inputs_per_detector * (max_n_inputs_per_detector - 1),
            device=self.device, generator=g
        )
        encoded_pairs_permutations = noise.argsort(dim=1, stable=True).to(dtype=torch.int32)

        # Create detector_anchors tensor
        self._detector_anchors = torch.zeros(
            self._n_detectors * self._n_anchors_per_detector * 2,
            dtype=torch.int32,
            device=self.device
        )

        self._lut_dm.initialize_detectors(
            encoded_pairs_permutations.flatten().contiguous(),
            max_n_inputs_per_detector,
            self._detector_anchors
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

    def _set_lookup_inidices_callback(self, cb):
        self._lookup_indices_callback = cb

    @staticmethod
    def forward_step(luts: List['LUTLayerBasic'], x, use_threads=False):
        if not (len(x.shape) == len(luts[0].input_shape()) + 2 and x.shape[2:] == luts[0].input_shape()):
            raise ValueError(
                f"Input x has invalid shape {x.shape}; expected {(x.shape[0], 1, *luts[0].input_shape())}"
                f" input_shape={luts[0].input_shape()}"
            )
        assert luts[0]._sequence_length == 1
        assert x.device == luts[0].device
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == 1, f"Input sequence_length {sequence_length} does not match constructor sequence_length 1"
        x = x.view(-1)
        output = torch.zeros([batch_size * luts[0]._n_outputs], dtype=torch.float32, device=x.device)

        args = []

        for lut in luts:
            lookup_indices = torch.zeros(
                [batch_size * lut._n_detectors], dtype=torch.int32, device=lut.device
            )
            min_anchor_deltas = torch.zeros(
                [batch_size * lut._n_detectors], dtype=torch.float32, device=lut.device, requires_grad=False
            )
            min_anchor_delta_indices = torch.zeros(
                [batch_size * lut._n_detectors], dtype=torch.int32, device=lut.device
            )

            sparse_firing_buffer = None
            if not lut._is_fully_connected:
                sparse_buffer_numel = (
                    1 + 2 * lut._n_detectors * lut._lut_dm.get_max_forward_groups_per_neuron() * batch_size
                ) * 4
                sparse_firing_buffer = lut._shared_context.get_sparse_firing_buffer(
                    sparse_buffer_numel,
                    lut.device,
                    lut._multi_id
                )

            stream_handles_tensor = None
            if lut.device.type == 'cuda':
                stream_handles_tensor = lut._shared_context.get_cuda_streams(lut.device, lut._multi_id)

            args.append((
                lut._weights, batch_size, x, lut._detector_anchors,
                output, lookup_indices, min_anchor_deltas, min_anchor_delta_indices,
                sparse_firing_buffer, stream_handles_tensor,
            ))

        if use_threads:
            threads = []
            # Start all threads
            for lut, lut_args in zip(luts, args):
                thread = threading.Thread(target=lut._lut_dm.forward_step, args=lut_args)
                threads.append(thread)

            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            for lut, lut_args in zip(luts, args):
                lut._lut_dm.forward_step(*lut_args)
                if lut._lookup_indices_callback is not None:
                    lut._lookup_indices_callback(lut_args[5], lut_args[6], lut_args[7])

        output = output.view((batch_size, 1) + luts[0].output_shape())

        first_lut = luts[0]
        if len(luts) == 1:
            return (
                output,
                args[0][5].view(batch_size, 1, first_lut._n_detectors),
                args[0][6].view(batch_size, 1, first_lut._n_detectors),
                args[0][7].view(batch_size, 1, first_lut._n_detectors)
            )
        else:
            return (
                output,
                [a[5].view(batch_size, 1, first_lut._n_detectors) for a in args],
                [a[6].view(batch_size, 1, first_lut._n_detectors) for a in args],
                [a[7].view(batch_size, 1, first_lut._n_detectors) for a in args]
            )

    @staticmethod
    def forward_step_concat(luts: List['LUTLayerBasic'], x, use_threads=False):
        if not (len(x.shape) == len(luts[0].input_shape()) + 2 and x.shape[2:] == luts[0].input_shape()):
            raise ValueError(
                f"Input x has invalid shape {x.shape}; expected {(x.shape[0], 'S', *luts[0].input_shape())} or {(x.shape[0], *luts[0].input_shape())}"
                f"where input_shape={luts[0].input_shape()}"
            )
        assert x.device == luts[0].device
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == luts[0]._sequence_length, f"Input sequence_length {sequence_length} does not match constructor sequence_length {luts[0]._sequence_length}"
        expected_shape = (batch_size, sequence_length) + luts[0].input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.view(-1)
        output = torch.zeros([batch_size * sequence_length * luts[0]._n_outputs], dtype=torch.float32, device=x.device)

        args = []

        for lut in luts:
            lookup_indices = torch.zeros(
                [batch_size * sequence_length * lut._n_detectors], dtype=torch.int32, device=lut.device
            )
            min_anchor_deltas = torch.zeros(
                [batch_size * sequence_length * lut._n_detectors], dtype=torch.float32,
                device=lut.device, requires_grad=False
            )
            min_anchor_delta_indices = torch.zeros(
                [batch_size * sequence_length * lut._n_detectors], dtype=torch.int32, device=lut.device
            )
            positional_lookup_indices = torch.zeros(
                [sequence_length * lut._n_detectors], dtype=torch.int32, device=lut.device
            )
            positional_min_deltas = torch.zeros(
                [sequence_length * lut._n_detectors], dtype=torch.float32,
                device=lut.device, requires_grad=False
            )
            positional_min_delta_indices = torch.zeros(
                [sequence_length * lut._n_detectors], dtype=torch.int32, device=lut.device
            )

            # TODO do not store this in context (calculate on the fly using buffer from shared context)
            firing_stat = torch.zeros(
                lut._n_lookup_neurons * batch_size * sequence_length,
                dtype=torch.float32, requires_grad=False,
                device=lut.device
            )

            sparse_buffer_numel = (1 + lut._n_lookup_neurons * batch_size * sequence_length) * 4
            sparse_firing_buffer = lut._shared_context.get_sparse_firing_buffer(
                sparse_buffer_numel,
                lut.device,
                lut._multi_id
            )

            stream_handles_tensor = None
            if lut.device.type == 'cuda':
                stream_handles_tensor = lut._shared_context.get_cuda_streams(lut.device, lut._multi_id)

            args.append((
                lut._weights,
                lut._positional_embeddings,
                batch_size, x,
                lut._detector_anchors,
                output,
                lookup_indices,
                min_anchor_deltas,
                min_anchor_delta_indices,
                positional_lookup_indices,
                positional_min_deltas,
                positional_min_delta_indices,
                sparse_firing_buffer,
                firing_stat,
                stream_handles_tensor
            ))

        if use_threads:
            threads = []
            # Start all threads
            for lut, lut_args in zip(luts, args):
                thread = threading.Thread(target=lut._lut_dm.forward_step_concat, args=lut_args)
                threads.append(thread)

            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            for lut, lut_args in zip(luts, args):
                lut._lut_dm.forward_step_concat(*lut_args)
                if lut._lookup_indices_callback is not None:
                    lut._lookup_indices_callback(lut_args[6], lut_args[7], lut_args[8])

        output = output.view((batch_size, sequence_length) + luts[0].output_shape())

        if len(luts) == 1:
            first_lut = luts[0]
            return (
                output,
                args[0][6].view(batch_size, sequence_length, first_lut._n_detectors),
                args[0][7].view(batch_size, sequence_length, first_lut._n_detectors),
                args[0][8].view(batch_size, sequence_length, first_lut._n_detectors),
                args[0][9].view(sequence_length, first_lut._n_detectors),
                args[0][10].view(sequence_length, first_lut._n_detectors),
                args[0][11].view(sequence_length, first_lut._n_detectors),
                args[0][13]
            )
        else:
            return (
                output,
                [a[6].view(batch_size, sequence_length, luts[i]._n_detectors) for i, a in enumerate(args)],
                [a[7].view(batch_size, sequence_length, luts[i]._n_detectors) for i, a in enumerate(args)],
                [a[8].view(batch_size, sequence_length, luts[i]._n_detectors) for i, a in enumerate(args)],
                [a[9].view(sequence_length, luts[i]._n_detectors) for i, a in enumerate(args)],
                [a[10].view(sequence_length, luts[i]._n_detectors) for i, a in enumerate(args)],
                [a[11].view(sequence_length, luts[i]._n_detectors) for i, a in enumerate(args)],
                [a[13] for a in args]
            )

    @staticmethod
    def backward_step(
        luts: List['LUTLayerBasic'], x, grad_output,
        lookup_indices_list, min_anchor_deltas_list, min_anchor_delta_indices_list,
        use_threads=False
    ):
        assert luts[0]._sequence_length == 1
        assert x.device == luts[0].device
        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == 1, f"Input sequence_length {sequence_length} does not match constructor sequence_length 1"
        expected_shape = (batch_size, 1) + luts[0].input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.view(-1)
        x_grad = torch.zeros_like(x)
        assert grad_output.device == luts[0].device
        assert grad_output.shape == (batch_size, 1) + luts[0].output_shape()
        grad_output = grad_output.view(-1)

        # Handle single vs multiple LUTs
        if len(luts) == 1:
            lookup_indices_list = [lookup_indices_list]
            min_anchor_deltas_list = [min_anchor_deltas_list]
            min_anchor_delta_indices_list = [min_anchor_delta_indices_list]

        args = []

        for i, lut in enumerate(luts):
            lookup_indices = lookup_indices_list[i]
            min_anchor_deltas = min_anchor_deltas_list[i]
            min_anchor_delta_indices = min_anchor_delta_indices_list[i]

            assert lookup_indices.device == lut.device
            assert lookup_indices.shape == (batch_size, 1, lut._n_detectors)
            lookup_indices = lookup_indices.view(-1)
            assert min_anchor_deltas.device == lut.device
            assert min_anchor_deltas.shape == (batch_size, 1, lut._n_detectors)
            min_anchor_deltas = min_anchor_deltas.view(-1)
            assert min_anchor_delta_indices.device == lut.device
            assert min_anchor_delta_indices.shape == (batch_size, 1, lut._n_detectors)
            min_anchor_delta_indices = min_anchor_delta_indices.view(-1)

            if lut._weights_gradient_policy.type == GradientType.Internal:
                target_w_grad = None
            elif lut._weights_gradient_policy.type == GradientType.Sparse:
                numel = ((lut._weights.numel() + 3) // 4) * 4
                target_w_grad = lut._shared_context.get_weight_gradients_buffer(
                    numel,
                    lut.device,
                    lut._multi_id
                )
            else:
                target_w_grad = torch.zeros_like(lut._weights, requires_grad=False)

            before_detectors_gradients = lut._shared_context.get_before_detectors_gradients_buffer(
                lut._n_lookup_neurons * batch_size,
                lut.device,
                lut._multi_id
            )

            sparse_firing_buffer = None
            if not lut._is_fully_connected:
                sparse_buffer_numel = (
                    1 + 2 * lut._n_detectors * lut._lut_dm.get_max_forward_groups_per_neuron() * batch_size
                ) * 4
                sparse_firing_buffer = lut._shared_context.get_sparse_firing_buffer(
                    sparse_buffer_numel,
                    lut.device,
                    lut._multi_id
                )

            if lut._weights_gradient_policy.type == GradientType.Internal:
                if lut._external_lr_hook is None:
                    raise ValueError("external_learning_rate_hook must be set when using GradientPolicy.Internal")
                external_lr = lut._external_lr_hook(lut._weights)
            else:
                external_lr = -1.0

            stream_handles_tensor = None
            if lut.device.type == 'cuda':
                stream_handles_tensor = lut._shared_context.get_cuda_streams(lut.device, lut._multi_id)

            args.append((
                lut._weights,
                batch_size,
                grad_output,
                x,
                lut._detector_anchors,
                lookup_indices,
                min_anchor_deltas,
                min_anchor_delta_indices,
                before_detectors_gradients,
                x_grad,
                external_lr,
                sparse_firing_buffer,
                target_w_grad,
                stream_handles_tensor
            ))

        if use_threads:
            threads = []
            # Start all threads
            for lut, lut_args in zip(luts, args):
                thread = threading.Thread(target=lut._lut_dm.backward_backprop, args=lut_args)
                threads.append(thread)

            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            for lut, lut_args in zip(luts, args):
                lut._lut_dm.backward_backprop(*lut_args[:14])

        # Post-process weight gradients
        processed_weight_grads = []
        for lut, lut_args in zip(luts, args):
            target_w_grad = lut_args[12]
            if lut._weights_gradient_policy.type == GradientType.Sparse:
                converter = lut._shared_context.get_dense_to_sparse_converter(lut._multi_id)
                n_weights_per_neuron = lut._n_outputs if lut._is_fully_connected else lut._lut_dm.get_max_forward_groups_per_neuron() * lut._forward_group_size
                indices, values = converter.dense_to_sparse_32(
                    target_w_grad, erase_input=True,
                    densify_buffers=lut._shared_context.get_densify_buffers(
                        lut._n_detectors * 2 * n_weights_per_neuron * batch_size,
                        lut.device,
                        lut._multi_id
                    )
                )
                if indices is not None:
                    if lut._weights_gradient_policy.normalized:
                        values /= values.abs().max().clip(1e-16)
                    target_w_grad = torch.sparse_coo_tensor(
                        indices=indices.unsqueeze(0),
                        values=values,
                        size=lut._weights.shape,
                        device=lut.device,
                        check_invariants=False,
                        is_coalesced=True,
                        requires_grad=False
                    )
                else:
                    target_w_grad = None
            elif lut._weights_gradient_policy.type == GradientType.Dense and lut._weights_gradient_policy.normalized:
                target_w_grad /= target_w_grad.abs().max().clip(1e-16)
            processed_weight_grads.append(target_w_grad)

        if len(luts) == 1:
            return x_grad.view(source_x_shape), processed_weight_grads[0]
        else:
            return x_grad.view(source_x_shape), processed_weight_grads

    @staticmethod
    def backward_step_concat(
        luts: List['LUTLayerBasic'], x, grad_output,
        lookup_indices_list, min_anchor_deltas_list, min_anchor_delta_indices_list,
        positional_lookup_indices_list, positional_min_deltas_list, positional_min_delta_indices_list,
        firing_stat_list,
        use_threads=False
    ):
        assert x.device == luts[0].device
        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == luts[0]._sequence_length, f"Input sequence_length {sequence_length} does not match constructor sequence_length {luts[0]._sequence_length}"
        expected_shape = (batch_size, sequence_length) + luts[0].input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.view(-1)
        x_grad = torch.zeros_like(x)

        assert grad_output.device == lut.device
        assert grad_output.shape == (batch_size, sequence_length) + lut.output_shape()

        grad_output = grad_output.view(-1)

        args = []

        for i, lut in enumerate(luts):
            lookup_indices = lookup_indices_list[i]
            min_anchor_deltas = min_anchor_deltas_list[i]
            min_anchor_delta_indices = min_anchor_delta_indices_list[i]
            positional_lookup_indices = positional_lookup_indices_list[i]
            positional_min_deltas = positional_min_deltas_list[i]
            positional_min_delta_indices = positional_min_delta_indices_list[i]
            firing_stat = firing_stat_list[i]

            assert lookup_indices.device == lut.device
            assert lookup_indices.shape == (batch_size, sequence_length, lut._n_detectors)
            lookup_indices = lookup_indices.view(-1)
            assert min_anchor_deltas.device == lut.device
            assert min_anchor_deltas.shape == (batch_size, sequence_length, lut._n_detectors)
            min_anchor_deltas = min_anchor_deltas.view(-1)
            assert min_anchor_delta_indices.device == lut.device
            assert min_anchor_delta_indices.shape == (batch_size, sequence_length, lut._n_detectors)
            min_anchor_delta_indices = min_anchor_delta_indices.view(-1)

            assert positional_lookup_indices.device == lut.device
            assert positional_lookup_indices.shape == (sequence_length, lut._n_detectors)
            positional_lookup_indices = positional_lookup_indices.view(-1)
            assert positional_min_deltas.device == lut.device
            assert positional_min_deltas.shape == (sequence_length, lut._n_detectors)
            positional_min_deltas = positional_min_deltas.view(-1)
            assert positional_min_delta_indices.device == lut.device
            assert positional_min_delta_indices.shape == (sequence_length, lut._n_detectors)
            positional_min_delta_indices = positional_min_delta_indices.view(-1)

            assert firing_stat.device == lut.device
            assert firing_stat.dtype == torch.float32
            assert firing_stat.shape == (lut._n_lookup_neurons * batch_size * sequence_length,)

            if lut._weights_gradient_policy.type == GradientType.Internal:
                target_w_grad = None
            elif lut._weights_gradient_policy.type == GradientType.Sparse:
                numel = ((lut._weights.numel() + 3) // 4) * 4
                target_w_grad = lut._shared_context.get_weight_gradients_buffer(
                    numel,
                    lut.device,
                    lut._multi_id
                )
            else:
                target_w_grad = torch.zeros_like(lut._weights, requires_grad=False)
            positional_grad = torch.zeros_like(lut._positional_embeddings, requires_grad=False)

            sparse_buffer_numel = (1 + lut._n_lookup_neurons * batch_size * sequence_length) * 4
            sparse_firing_buffer = lut._shared_context.get_sparse_firing_buffer(
                sparse_buffer_numel,
                lut.device,
                lut._multi_id
            )

            if lut._weights_gradient_policy.type == GradientType.Internal:
                if lut._external_lr_hook is None:
                    raise ValueError("external_learning_rate_hook must be set when using GradientPolicy.Internal")
                external_lr = lut._external_lr_hook(lut._weights)
            else:
                external_lr = -1.0

            stream_handles_tensor = None
            if lut.device.type == 'cuda':
                stream_handles_tensor = lut._shared_context.get_cuda_streams(lut.device, lut._multi_id)

            args.append((
                lut._weights,
                lut._positional_embeddings,
                batch_size,
                grad_output,
                x,
                lut._detector_anchors,
                lookup_indices,
                min_anchor_deltas,
                min_anchor_delta_indices,
                positional_lookup_indices,
                positional_min_deltas,
                positional_min_delta_indices,
                sparse_firing_buffer,
                firing_stat,
                x_grad, positional_grad,
                external_lr,
                target_w_grad,
                stream_handles_tensor,
            ))

        if use_threads:
            threads = []
            # Start all threads
            for lut, lut_args in zip(luts, args):
                thread = threading.Thread(target=lut._lut_dm.backward_backprop_concat, args=lut_args[:19])
                threads.append(thread)

            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            for lut, lut_args in zip(luts, args):
                lut._lut_dm.backward_backprop_concat(*lut_args[:19])

        # Post-process weight and positional gradients
        processed_weight_grads = []
        processed_positional_grads = []
        for lut, lut_args in zip(luts, args):
            target_w_grad = lut_args[-2]
            positional_grad = lut_args[-4]
            
            if lut._weights_gradient_policy.type == GradientType.Sparse:
                converter = lut._shared_context.get_dense_to_sparse_converter(lut._multi_id)
                n_weights_per_neuron = lut._n_outputs if lut._is_fully_connected else lut._lut_dm.get_max_forward_groups_per_neuron() * lut._forward_group_size
                indices, values = converter.dense_to_sparse_32(
                    target_w_grad, erase_input=True,
                    densify_buffers=lut._shared_context.get_densify_buffers(
                        lut._n_detectors * n_weights_per_neuron * (lut._sequence_length * (lut._sequence_length - 1) / 2) * batch_size,
                        lut.device,
                        lut._multi_id
                    )
                )
                if indices is not None:
                    if lut._weights_gradient_policy.normalized:
                        values /= values.abs().max().clip(1e-16)
                    target_w_grad = torch.sparse_coo_tensor(
                        indices=indices.unsqueeze(0),
                        values=values,
                        size=lut._weights.shape,
                        device=lut.device,
                        check_invariants=False,
                        is_coalesced=True,
                        requires_grad=False
                    )
                else:
                    target_w_grad = None
            elif lut._weights_gradient_policy.type == GradientType.Dense and lut._weights_gradient_policy.normalized:
                target_w_grad /= target_w_grad.abs().max().clip(1e-16)

            if lut._weights_gradient_policy.normalized:
                positional_grad /= positional_grad.abs().max().clip(1e-16)

            processed_weight_grads.append(target_w_grad)
            processed_positional_grads.append(positional_grad)

        if len(luts) == 1:
            return x_grad.view(source_x_shape), processed_weight_grads[0], processed_positional_grads[0]
        else:
            return x_grad.view(source_x_shape), processed_weight_grads, processed_positional_grads

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
            x, _, __, lut_layer = args
            ctx.lut_layer = lut_layer
            if lut_layer._sequence_length == 1:
                (
                    output, lookup_indices, min_anchor_deltas, 
                    min_anchor_delta_indices
                ) = LUTLayerBasic.forward_step([lut_layer], x)
                ctx.save_for_backward(x, lookup_indices, min_anchor_deltas, min_anchor_delta_indices)
            else:
                (
                    output, lookup_indices, min_anchor_deltas, 
                    min_anchor_delta_indices, positional_lookup_indices, 
                    positional_min_deltas, positional_min_delta_indices, 
                    firing_stat
                ) = LUTLayerBasic.forward_step_concat([lut_layer], x)
                ctx.save_for_backward(
                    x, lookup_indices, min_anchor_deltas, min_anchor_delta_indices,
                    positional_lookup_indices, positional_min_deltas, 
                    positional_min_delta_indices, firing_stat
                )
            return output

        @staticmethod
        def backward(ctx, *grad_outputs):
            (grad_output,) = grad_outputs
            if ctx.lut_layer._sequence_length == 1:
                (x, lookup_indices, min_anchor_deltas, min_anchor_delta_indices) = ctx.saved_tensors
                x_grad, w_grad = LUTLayerBasic.backward_step(
                    [ctx.lut_layer], x, grad_output,
                    lookup_indices,
                    min_anchor_deltas,
                    min_anchor_delta_indices
                )
                return x_grad, w_grad, None, None
            else:
                (
                    x, lookup_indices, min_anchor_deltas, min_anchor_delta_indices, positional_lookup_indices,
                    positional_min_deltas, positional_min_delta_indices, firing_stat
                ) = ctx.saved_tensors
                x_grad, w_grad, pe_grad = LUTLayerBasic.backward_step_concat(
                    [ctx.lut_layer], x, grad_output,
                    lookup_indices,
                    min_anchor_deltas,
                    min_anchor_delta_indices,
                    positional_lookup_indices,
                    positional_min_deltas,
                    positional_min_delta_indices,
                    firing_stat
                )
                return x_grad, w_grad, pe_grad, None

    def forward(self, x):
        if self._sequence_length == 1:
            return LUTLayerBasic.LUTForwardFN.apply(x, self._weights, None, self)
        else:
            return LUTLayerBasic.LUTForwardFN.apply(x, self._weights, self._positional_embeddings, self)

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
        positional_dim=None,
        weights_gradient_policy: GradientPolicy = GradientPolicy(GradientType.Dense, normalized=False),
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
        _explicit_anchors=None,
        _int_rescaler=0.001,
        _forward_group_size=64,
        _backward_group_size=8,
        _max_groups_in_growth_buffer=2 ** 20,
        random_seed=1,
        device=None
    ):
        if receptive_field_shape is None:
            assert receptive_field_stride_shape is None
            receptive_field_shape = input_shape
            receptive_field_stride_shape = input_shape

        c_helper_1 = Conv2DSynapseGrowthHelper(
            input_shape[0], input_shape[1],
            receptive_field_shape[0], receptive_field_shape[1],
            receptive_field_stride_shape[0], receptive_field_stride_shape[1],
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
            positional_dim=positional_dim, weights_gradient_policy=weights_gradient_policy,
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
                input_ids=self.get_input_neuron_ids().view(input_shape) + 1,
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

            self.initialize_detectors(random_seed)
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
        positional_dim=None,
        weights_gradient_policy: GradientPolicy = GradientPolicy(GradientType.Dense, normalized=False),
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
        _explicit_anchors=None,
        _int_rescaler=0.001,
        _forward_group_size=64,
        _backward_group_size=8,
        _max_groups_in_growth_buffer=2 ** 20,
        random_seed=1,
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
            positional_dim=positional_dim,
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
    A module that runs multiple LUTLayerBasic instances in parallel or sequentially.
    All layers must have the same input_shape, output_shape, and sequence_length.
    Each layer is assigned a unique multi_id for stream isolation.
    """
    
    def __init__(self, layers: List[LUTLayerBasic], use_threads: bool = True):
        super().__init__()
        
        if len(layers) < 2:
            raise ValueError("MultiLUT requires at least two layers")
        
        # Validate all layers have compatible shapes
        first_layer = layers[0]
        input_shape = first_layer.input_shape()
        output_shape = first_layer.output_shape()
        sequence_length = first_layer.sequence_length()
        
        for i, layer in enumerate(layers):
            if not isinstance(layer, LUTLayerBasic):
                raise ValueError(f"Layer {i} is not an instance of LUTLayerBasic")
            if layer.input_shape() != input_shape:
                raise ValueError(f"Layer {i} has input_shape {layer.input_shape()}, expected {input_shape}")
            if layer.output_shape() != output_shape:
                raise ValueError(f"Layer {i} has output_shape {layer.output_shape()}, expected {output_shape}")
            if layer.sequence_length() != sequence_length:
                raise ValueError(f"Layer {i} has sequence_length {layer.sequence_length()}, expected {sequence_length}")
        
        # Assign multi_id to each layer
        for multi_id, layer in enumerate(layers):
            layer._multi_id = multi_id
        
        self.layers = nn.ModuleList(layers)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._sequence_length = sequence_length
        self._use_threads = use_threads
        self._all_weights = None
        self._all_positional_encodings = None
    
    def input_shape(self):
        return self._input_shape
    
    def output_shape(self):
        return self._output_shape
    
    def sequence_length(self):
        return self._sequence_length
    
    def set_external_learning_rate_hook(self, hook_fn):
        """Set the external learning rate hook for all layers."""
        for layer in self.layers:
            layer.set_external_learning_rate_hook(hook_fn)
    
    def forward(self, x):
        """Forward pass that runs all layers using static methods."""
        # Lazy fill weights and positional encodings if needed
        if self._all_weights is None:
            self._all_weights = [layer._weights for layer in self.layers]
        
        if self._sequence_length > 1:
            if self._all_positional_encodings is None:
                self._all_positional_encodings = [layer._positional_embeddings for layer in self.layers]
            return MultiLUT.MultiLUTForwardFN.apply(x, self, *self._all_weights, *self._all_positional_encodings)
        else:
            return MultiLUT.MultiLUTForwardFN.apply(x, self, *self._all_weights)
    
    class MultiLUTForwardFN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            """Run forward pass for all layers."""
            x, multi_lut = args[0], args[1]
            batch_size = x.shape[0]
            
            # Call static forward method
            if multi_lut._sequence_length == 1:
                result = LUTLayerBasic.forward_step(multi_lut.layers, x, use_threads=multi_lut._use_threads)
            else:
                result = LUTLayerBasic.forward_step_concat(multi_lut.layers, x, use_threads=multi_lut._use_threads)
            
            # Extract output and reshape
            output = result[0].view((batch_size, multi_lut._sequence_length) + multi_lut.output_shape())
            
            # Save full result for backward (same structure as returned by static methods)
            ctx.multi_lut = multi_lut
            ctx.save_for_backward(x)
            ctx.result = result
            
            return output

        @staticmethod
        def backward(ctx, *grad_outputs):
            """Run backward pass for all layers."""
            (grad_output,) = grad_outputs
            multi_lut = ctx.multi_lut
            x = ctx.saved_tensors[0]
            result = ctx.result
            
            # Use the same result structure directly
            if multi_lut._sequence_length == 1:
                # result is (output, lookup_indices, min_anchor_deltas, min_anchor_delta_indices)
                # or (output, [lookup_indices...], [min_anchor_deltas...], [min_anchor_delta_indices...])
                x_grad, weight_grads = LUTLayerBasic.backward_step(
                    multi_lut.layers, x, grad_output,
                    result[1], result[2], result[3],
                    use_threads=multi_lut._use_threads
                )
                
                # Return gradients: x_grad, None (for multi_lut), *weight_grads
                return (x_grad.view(x.shape), None) + tuple(weight_grads)
            else:
                # result is (output, lookup_indices, min_anchor_deltas, min_anchor_delta_indices, 
                #           positional_lookup_indices, positional_min_deltas, positional_min_delta_indices, firing_stat)
                # or (output, [lookup_indices...], [min_anchor_deltas...], [min_anchor_delta_indices...],
                #     [positional_lookup_indices...], [positional_min_deltas...], [positional_min_delta_indices...], [firing_stat...])
                x_grad, weight_grads, pe_grads = LUTLayerBasic.backward_step_concat(
                    multi_lut.layers, x, grad_output,
                    result[1], result[2], result[3],
                    result[4], result[5], result[6], result[7],
                    use_threads=multi_lut._use_threads
                )
                
                # Return gradients: x_grad, None (for multi_lut), *weight_grads, *pe_grads
                return (x_grad.view(x.shape), None) + tuple(weight_grads) + tuple(pe_grads)
    
    def to(self, *args, **kwargs):
        """Move the module to a different device/dtype. Resets cached weights and positional encodings."""
        result = super().to(*args, **kwargs)
        self._all_weights = None
        self._all_positional_encodings = None
        return result
    
    def __repr__(self):
        return f'MultiLUT({len(self.layers)} layers, input_shape={self.input_shape()}, output_shape={self.output_shape()}, sequence_length={self.sequence_length()})'
