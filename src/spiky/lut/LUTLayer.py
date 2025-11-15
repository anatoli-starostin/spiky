import torch
import torch.nn as nn
import torch.nn.functional as nf
from typing import List, Dict, Tuple, AnyStr
from dataclasses import dataclass

from spiky_cuda import LUTDataManagerF, LUTDataManagerI
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper
from spiky.util.chunk_of_connections import ChunkOfConnections


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


class LUTLayer(nn.Module):
    def __init__(
        self, n_inputs, n_outputs,
        n_detectors, n_anchors_per_detector,
        sequence_length,
        synapse_metas: List[SynapseMeta],
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _initial_synapse_capacity=None,
        _forward_group_size: int = 64,
        _backward_group_size: int = 64
    ):
        super().__init__()

        assert len(synapse_metas) > 0
        assert sequence_length > 0
        self.device = torch.device("cpu")
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs

        self._n_detectors = n_detectors
        self._n_anchors_per_detector = n_anchors_per_detector
        self._sequence_length = sequence_length

        if _initial_synapse_capacity is None:
            _initial_synapse_capacity = self._n_lookup_neurons * n_outputs

        if summation_dtype == torch.float32:
            self._lut_dm = LUTDataManagerF(
                n_inputs, n_outputs, n_detectors, n_anchors_per_detector,
                sequence_length,
                _initial_synapse_capacity,
                _forward_group_size,
                _backward_group_size
            )
        else:
            self._lut_dm = LUTDataManagerI(
                n_inputs, n_outputs, n_detectors, n_anchors_per_detector,
                sequence_length,
                _initial_synapse_capacity,
                _forward_group_size,
                _backward_group_size,
                _int_rescaler
            )

        self._n_lookup_neurons = self._lut_dm.get_number_of_lookup_neurons()

        for i, sm in enumerate(synapse_metas):
            m_id = self._lut_dm.register_synapse_meta(
                learning_rate=sm.learning_rate,
                min_synaptic_weight=sm.min_weight,
                max_synaptic_weight=sm.max_weight,
                initial_noise_level=sm.initial_noise_level,
                initial_weight=sm.initial_weight
            )
            assert m_id == i

        self._lut_dm.initialize_neurons()
        self._input_neuron_ids = torch.arange(0, self._n_inputs, dtype=torch.int32, device=self.device)
        self._detector_neuron_ids = torch.arange(0, self._n_detectors, dtype=torch.int32, device=self.device)
        self._lookup_neuron_ids = torch.arange(0, self._n_lookup_neurons, dtype=torch.int32, device=self.device)
        self._output_neuron_ids = torch.arange(self._n_lookup_neurons, self._n_lookup_neurons + n_outputs, dtype=torch.int32, device=self.device)
        self._weights = None
        self._last_w_grad = None

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

    def initialize_detectors(self):
        max_n_inputs_per_detector = self._lut_dm.calculate_max_n_inputs_per_detector()
        assert max_n_inputs_per_detector * (max_n_inputs_per_detector - 1) >= self._n_anchors_per_detector

        noise = torch.rand(
            self._n_detectors, max_n_inputs_per_detector * (max_n_inputs_per_detector - 1),
            device=self.device
        )
        encoded_pairs_permutations = noise.argsort(dim=1)

        self._lut_dm.initialize_detectors(
            encoded_pairs_permutations.flatten().contiguous(),
            max_n_inputs_per_detector
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
        return self._lut_dm.get_number_of_synapses()

    def input_shape(self):
        return (self._n_outputs,)

    def output_shape(self):
        return (self._n_outputs,)

    def __repr__(self):
        return f'LUTLayer({self.n_inputs()} inputs, {self.n_detectors()} detectors, {self.n_lookup_neurons()} lookup neurons, {self.n_outputs()} outputs, {self.n_synapses()} synapses, {self._lut_dm})'

    def add_lookup_connections(
        self, chunk_of_connections: ChunkOfConnections,
        ids_shift=0,
        random_seed: int = None
    ):
        self._lut_dm.add_lookup_connections(
            chunk_of_connections.get_connections(),
            chunk_of_connections.get_single_group_size(),
            ids_shift,
            random_seed
        )

    def compile_lut(self, shuffle_synapses_random_seed: int = None, _only_trainable_backwards=True):
        n_weights = self._lut_dm.get_weights_dimension()
        self._weights = nn.Parameter(torch.zeros([n_weights], dtype=torch.float32, device=self.device))
        self._lut_dm.compile(_only_trainable_backwards, self._weights.detach(), shuffle_synapses_random_seed)

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

    def forward_step(self, x):
        assert x.device == self.device
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == self._sequence_length, f"Input sequence_length {sequence_length} does not match constructor sequence_length {self._sequence_length}"
        expected_shape = (batch_size, sequence_length) + self.input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.flatten().contiguous()

        output = torch.zeros([batch_size * sequence_length * self._n_outputs], dtype=torch.float32, device=self.device)

        lookup_indices = torch.zeros([batch_size * sequence_length * self._n_detectors], dtype=torch.int32, device=self.device)
        min_anchor_deltas = torch.zeros([batch_size * sequence_length * self._n_detectors], dtype=torch.float32, device=self.device)
        min_anchor_deltas_indices = torch.zeros([batch_size * sequence_length * self._n_detectors], dtype=torch.int32, device=self.device)

        self._lut_dm.forward_step(
            self._weights,
            batch_size, sequence_length, x,
            output,
            lookup_indices,
            min_anchor_deltas,
            min_anchor_deltas_indices,
        )

        return (
            output.reshape((batch_size, sequence_length) + self.output_shape()),
            lookup_indices.reshape(batch_size, sequence_length, self._n_detectors),
            min_anchor_deltas.reshape(batch_size, sequence_length, self._n_detectors),
            min_anchor_deltas_indices.reshape(batch_size, sequence_length, self._n_detectors),
        )

    def backward_step(
        self, x, grad_output,
        lookup_indices, min_anchor_deltas, min_anchor_deltas_indices
    ):
        assert x.device == self.device
        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == self._sequence_length, f"Input sequence_length {sequence_length} does not match constructor sequence_length {self._sequence_length}"
        expected_shape = (batch_size, sequence_length) + self.input_shape()
        assert x.shape == expected_shape, f"Expected input shape {expected_shape}, got {x.shape}"

        x = x.flatten().contiguous()
        assert lookup_indices.device == self.device
        assert lookup_indices.shape == (batch_size, sequence_length, self._n_detectors)
        lookup_indices = lookup_indices.flatten().contiguous()
        assert min_anchor_deltas.device == self.device
        assert min_anchor_deltas.shape == (batch_size, sequence_length, self._n_detectors)
        min_anchor_deltas = min_anchor_deltas.flatten().contiguous()
        assert min_anchor_deltas_indices.device == self.device
        assert min_anchor_deltas_indices.shape == (batch_size, sequence_length, self._n_detectors)
        min_anchor_deltas_indices = min_anchor_deltas_indices.flatten().contiguous()

        x_grad = torch.zeros_like(x)
        self._last_w_grad = torch.zeros_like(self._weights)

        assert grad_output.device == self.device
        assert grad_output.shape == (batch_size, sequence_length) + self.output_shape()

        grad_output = grad_output.flatten().contiguous()
        self._lut_dm.backward_backprop(
            self._weights,
            batch_size, sequence_length,
            grad_output,
            x, lookup_indices,
            min_anchor_deltas,
            min_anchor_deltas_indices,
            x_grad, self._last_w_grad
        )

        with torch.no_grad():
            m = self._last_w_grad.abs().max()
            if m < 1e-16:
                m = 1e-16
            self._last_w_grad /= m

        return x_grad.reshape(source_x_shape), self._last_w_grad

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
        return self

    class LUTForwardFN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            x, _, lut_layer = args
            ctx.lut_layer = lut_layer
            output, lookup_indices, min_anchor_deltas, min_anchor_deltas_indices = lut_layer.forward_step(x)
            ctx.save_for_backward(x, lookup_indices, min_anchor_deltas, min_anchor_deltas_indices)
            return output

        @staticmethod
        def backward(ctx, *grad_outputs):
            (grad_output,) = grad_outputs
            (x, lookup_indices, min_anchor_deltas, min_anchor_deltas_indices) = ctx.saved_tensors

            x_grad, w_grad = ctx.lut_layer.backward_step(
                x, grad_output,
                lookup_indices,
                min_anchor_deltas,
                min_anchor_deltas_indices,
            )
            return x_grad, w_grad, None

    def forward(self, x):
        return LUTLayer.LUTForwardFN.apply(x, self._weights, self)

    def _count_synapses(self, neuron_ids: torch.Tensor, forward_or_backward: True):
        return self._lut_dm.count_synapses(neuron_ids, forward_or_backward)

    def _export_synapses(
        self, neuron_ids: torch.Tensor,
        source_ids: torch.Tensor,
        weights: torch.Tensor,
        target_ids: torch.Tensor,
        forward_or_backward: True,
        synapse_metas: torch.Tensor = None
    ):
        self._lut_dm.export_synapses(
            self._weights,
            neuron_ids,
            source_ids,
            weights,
            target_ids,
            forward_or_backward,
            synapse_metas
        )


class Conv2DLUTLayer(LUTLayer):
    def __init__(
        self, input_shape,
        n_anchors_per_detector,
        detectors_shape,
        output_kernel_shape,
        sequence_length,
        receptive_field_shape=None,
        receptive_field_stride_shape=None,
        lut_receptive_field_shape=None,
        lut_receptive_field_stride_shape=None,
        synapse_meta=SynapseMeta(),
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _forward_group_size: int = 64,
        _backward_group_size: int = 64,
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

        lut_shape = c_helper_1.out_h, c_helper_1.out_h
        n_lut_channels = n_anchors_per_detector ** 2

        if lut_receptive_field_shape is None:
            assert lut_receptive_field_stride_shape is None
            lut_receptive_field_shape = lut_shape
            lut_receptive_field_stride_shape = lut_shape

        c_helper_2 = Conv2DSynapseGrowthHelper(
            lut_shape[0], lut_shape[1],
            lut_receptive_field_shape[0], lut_receptive_field_shape[1],
            lut_receptive_field_stride_shape[0], lut_receptive_field_stride_shape[1],
            output_kernel_shape[0], output_kernel_shape[1],
            n_input_channels=n_lut_channels
        )

        n_inputs = input_shape[0] * input_shape[1]
        n_outputs = c_helper_2.out_h * c_helper_2.out_w
        n_detectors = lut_shape[0] * lut_shape[1]

        super().__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_detectors=n_detectors,
            n_anchors_per_detector=n_anchors_per_detector,
            sequence_length=sequence_length,
            synapse_metas=[synapse_meta],
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            _initial_synapse_capacity=c_helper_2.n_connections(),
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size
        )

        if device is not None:
            self.to(device=device)
        else:
            device = torch.device("cpu")

        self.add_detector_connections(
            chunk_of_connections=c_helper_1.grow_synapses(
                input_ids=self.get_input_neuron_ids().reshape(input_shape) + 1,
                output_ids=self.get_detector_neuron_ids().reshape(detectors_shape[0], detectors_shape[1]) + 1,
                device=device,
                seed=random_seed
            ),
            ids_shift=-1,
            random_seed=random_seed
        )

        self.initialize_detectors()

        self.add_connections(
            chunk_of_connections=c_helper_2.grow_synapses(
                input_ids=self.get_lookup_neuron_ids().reshape(lut_shape) + 1,
                output_ids=self.get_output_neuron_ids().reshape(c_helper_2.out_h, c_helper_2.out_w) + 1,
                device=device,
                seed=random_seed
            ),
            ids_shift=-1,
            random_seed=random_seed
        )

        self.compile_lut()
        self._input_shape = input_shape
        self._output_shape = (c_helper_2.out_h, c_helper_2.out_w)
        self._lut_receptive_field_shape = lut_receptive_field_shape

    def forward(self, x):
        if x.shape == (x.shape[0],) + self._input_shape:
            x = x.unsqueeze(1)
        elif not (len(x.shape) == len(self._input_shape) + 2 and x.shape[2:] == self._input_shape):
            raise ValueError(
                f"Input x has invalid shape {x.shape}; expected {(x.shape[0], 'S', *self._input_shape)} or {(x.shape[0], *self._input_shape)}"
                f"where input_shape={self._input_shape}"
            )

        result = super().forward(x)
        if x.shape == (x.shape[0],) + self._input_shape:
            result = result.squeeze(1)
        return result

    def input_shape(self):
        return self._input_shape

    def output_shape(self):
        return self._output_shape

    def lut_receptive_field_shape(self):
        return self._lut_receptive_field_shape

    def export_weights(self):
        n_synapses = self.n_synapses()
        source_ids = torch.zeros([n_synapses], dtype=torch.int32, device=self.device)
        target_ids = torch.zeros([n_synapses], dtype=torch.int32, device=self.device)
        weights = torch.zeros([n_synapses], dtype=torch.float32, device=self.device)

        self._export_synapses(
            self.get_output_neuron_ids(),
            source_ids,
            weights,
            target_ids,
            forward_or_backward=False
        )

        order = torch.argsort(source_ids, stable=True, descending=False)
        order = order[torch.argsort(target_ids[order], stable=True, descending=False)]

        return weights[order].reshape(self._output_shape + self._lut_receptive_field_shape)
