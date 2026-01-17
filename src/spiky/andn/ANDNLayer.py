import torch
import torch.nn as nn
import torch.nn.functional as nf
from typing import List, Dict, Tuple, AnyStr
from dataclasses import dataclass

from spiky_cuda import ANDNDataManagerF, ANDNDataManagerI
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper, InhibitionGrid2DHelper, RandomInhibition2DHelper
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


class ANDNLayer(nn.Module):
    def __init__(
        self, n_inputs, n_outputs,
        n_detectors, max_inputs_per_detector,
        synapse_metas: List[SynapseMeta],
        spiking_inhibition=True,
        anti_hebb_coeff=0.0,
        backprop_hebb_ratio_on_torch_backward=1.0,
        relu_output=False,
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _initial_synapse_capacity=None,
        _forward_group_size: int = 64,
        _backward_group_size: int = 64
    ):
        super().__init__()

        assert len(synapse_metas) > 0
        self.device = torch.device("cpu")
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs

        self._n_detectors = n_detectors
        self._spiking_inhibition = spiking_inhibition
        self._max_inputs_per_detector = max_inputs_per_detector
        self._anti_hebb_coeff = anti_hebb_coeff

        self._descendant_andn_layer = None
        self._backprop_hebb_ratio_on_torch_backward = backprop_hebb_ratio_on_torch_backward
        self._relu_output = relu_output

        if _initial_synapse_capacity is None:
            _initial_synapse_capacity = n_inputs * n_outputs

        dm_args = (
            n_inputs, n_outputs,
            _initial_synapse_capacity,
            _forward_group_size,
            _backward_group_size,
            spiking_inhibition
        )
        if summation_dtype == torch.float32:
            self._andn_dm = ANDNDataManagerF(*dm_args)
        else:
            self._andn_dm = ANDNDataManagerI(*dm_args, _int_rescaler)

        for i, sm in enumerate(synapse_metas):
            m_id = self._andn_dm.register_synapse_meta(
                learning_rate=sm.learning_rate,
                min_synaptic_weight=sm.min_weight,
                max_synaptic_weight=sm.max_weight,
                initial_noise_level=sm.initial_noise_level,
                initial_weight=sm.initial_weight
            )
            assert m_id == i

        self._andn_dm.initialize_neurons()
        self._input_neuron_ids = torch.arange(0, n_inputs, dtype=torch.int32, device=self.device)
        self._output_neuron_ids = torch.arange(n_inputs, n_inputs + n_outputs, dtype=torch.int32, device=self.device)
        self._empty_int_tensor = torch.tensor([], dtype=torch.int32)
        self._empty_float_tensor = torch.tensor([], dtype=torch.float32)
        self._weights = None
        self._hebbian_history = []
        self._last_w_grad = None
        self._n_hebbian_ancestors = 0

    def initialize_detectors(self, detectors):
        assert self._n_detectors > 0
        assert detectors.device == self.device
        assert detectors.shape == (self._n_detectors, self._max_inputs_per_detector)
        # detectors tensor must contain input neuron indices for each detector
        # it has shape [n_detectors, max_inputs_per_detector], -1-s are ignored
        assert (detectors != -1).sum(dim=-1).min() >= 2
        self._andn_dm.initialize_detectors(detectors.flatten().contiguous(), self._max_inputs_per_detector)

    def get_input_neuron_ids(self):
        return self._input_neuron_ids

    def get_output_neuron_ids(self):
        return self._output_neuron_ids

    def n_input_neurons(self):
        return self._n_inputs

    def n_output_neurons(self):
        return self._n_outputs

    def n_detectors(self):
        return self._n_detectors

    def n_synapses(self):
        return self._andn_dm.get_number_of_synapses()

    def output_shape(self):
        return (self._n_outputs,)

    def __repr__(self):
        return f'ANDNLayer({self.n_input_neurons()} inputs, {self.n_detectors()} detectors, {self.n_output_neurons()} outputs, {self.n_synapses()} synapses, {self._andn_dm})'

    def add_connections(
        self, chunk_of_connections: ChunkOfConnections,
        ids_shift=0,
        random_seed: int = None
    ):
        self._andn_dm.add_connections(
            chunk_of_connections.get_connections(),
            chunk_of_connections.get_single_group_size(),
            ids_shift,
            random_seed
        )

    def compile_andn(self, shuffle_synapses_random_seed: int = None, _only_trainable_backwards=True):
        n_weights = self._andn_dm.get_weights_dimension()
        self._weights = nn.Parameter(torch.zeros([n_weights], dtype=torch.float32, device=self.device))
        self._andn_dm.compile(_only_trainable_backwards, self._weights.detach(), shuffle_synapses_random_seed)
        self._andn_dm.to_device(-1)
        if self.device.type == 'cuda':
            self._andn_dm.to_device(self.device.index)

    def get_smallest_distinguishable_fraction(self) -> float:
        return self._andn_dm.get_smallest_distinguishable_fraction()

    def get_epsilon(self) -> float:
        return self._andn_dm.get_epsilon()

    def get_summation_type(self) -> torch.dtype:
        t = self._andn_dm.get_summations_data_type()
        if t == 'int32':
            return torch.int32
        elif t == 'float32':
            return torch.float32
        else:
            raise RuntimeError('Unsupported summations type: ' + t)

    def get_memory_stats(self) -> str:
        return self._andn_dm.get_memory_stats()

    def get_profiling_stats(self) -> str:
        return self._andn_dm.get_profiling_stats()

    def reset_profiler(self):
        self._andn_dm.reset_profiler()

    def set_descendant_andn_layer(self, descendant_andn_layer):
        assert isinstance(descendant_andn_layer, (ANDNLayer, InhibitionLayer,))
        assert descendant_andn_layer._n_inputs == self._n_outputs
        self._descendant_andn_layer = descendant_andn_layer
        descendant_andn_layer._n_hebbian_ancestors += 1

    def detach_descendant_andn_layer(self):
        assert self._descendant_andn_layer is not None
        self._descendant_andn_layer._n_hebbian_ancestors -= 1
        self._descendant_andn_layer = None

    def forward_step(self, x):
        assert x.device == self.device
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self._n_inputs).flatten().contiguous()

        output = torch.zeros([batch_size * self._n_outputs], dtype=torch.float32, device=self.device)
        if self._n_detectors > 0:
            input_winner_ids = torch.zeros([batch_size * self._n_detectors], dtype=torch.int32, device=self.device)
            input_prewinner_ids = torch.zeros([batch_size * self._n_detectors], dtype=torch.int32, device=self.device)
            input_winning_stat = torch.zeros([batch_size * self._n_inputs], dtype=torch.int32, device=self.device)
        else:
            input_winner_ids = self._empty_int_tensor
            input_prewinner_ids = self._empty_int_tensor
            input_winning_stat = self._empty_int_tensor

        self._andn_dm.forward(
            self._weights.detach(),
            batch_size, x,
            output,
            input_winner_ids,
            input_prewinner_ids,
            input_winning_stat
        )

        if self._relu_output:
            output = nf.relu(output)

        if self._n_detectors > 0:
            return (
                output.reshape((batch_size,) + self.output_shape()),
                input_winner_ids.reshape(batch_size, self._n_detectors),
                input_prewinner_ids.reshape(batch_size, self._n_detectors),
                input_winning_stat.reshape(batch_size, self._n_inputs)
            )
        else:
            return (
                output.reshape((batch_size,) + self.output_shape()),
                input_winner_ids,
                input_prewinner_ids,
                input_winning_stat
            )

    def backward_step(
        self, x, grad_output,
        input_winner_ids, input_prewinner_ids, input_winning_stat,
        output, output_winner_ids, output_prewinner_ids, output_winning_stat,
        bh_ratio=1.0
    ):
        assert x.device == self.device
        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        x = x.flatten().contiguous()
        if self._n_detectors > 0:
            assert input_winning_stat.device == self.device
            assert input_winning_stat.shape == (batch_size, self._n_inputs)
            input_winning_stat = input_winning_stat.flatten().contiguous()
            assert input_winner_ids.device == self.device
            assert input_winner_ids.shape == (batch_size, self._n_detectors)
            assert input_prewinner_ids.device == self.device
            assert input_prewinner_ids.shape == (batch_size, self._n_detectors)
            input_winner_ids = input_winner_ids.flatten().contiguous()
            input_prewinner_ids = input_prewinner_ids.flatten().contiguous()

        x_grad = torch.zeros_like(x)
        self._last_w_grad = None

        if bh_ratio < 1.0:
            self._last_w_grad = torch.zeros_like(self._weights)

            assert grad_output.device == self.device
            assert grad_output.shape == (batch_size,) + self.output_shape()

            grad_output = grad_output.flatten().contiguous()
            self._andn_dm.backward_backprop(
                self._weights.detach(),
                batch_size,
                grad_output,
                x, input_winner_ids,
                input_prewinner_ids,
                input_winning_stat,
                x_grad, self._last_w_grad
            )

            with torch.no_grad():
                m = self._last_w_grad.abs().max()
                if m < 1e-16:
                    m = 1e-16
                self._last_w_grad *= (1.0 - bh_ratio) / m

        if bh_ratio > 0.0:
            assert self._descendant_andn_layer is not None
            assert output.device == self.device
            assert output.shape == (batch_size,) + self.output_shape()
            output = output.flatten().contiguous()
            assert output_winner_ids.device == self.device
            assert output_winner_ids.shape == (batch_size, self._descendant_andn_layer._n_detectors)
            assert output_prewinner_ids.device == self.device
            assert output_prewinner_ids.shape == (batch_size, self._descendant_andn_layer._n_detectors)
            assert output_winning_stat.device == self.device
            assert output_winning_stat.shape == (batch_size, self._n_outputs)
            output_winner_ids = output_winner_ids.flatten().contiguous()
            output_prewinner_ids = output_prewinner_ids.flatten().contiguous()
            output_winning_stat = output_winning_stat.flatten().contiguous()

            w_grad_hebb = torch.zeros_like(self._weights)

            self._andn_dm.backward_hebb(
                self._weights.detach(),
                batch_size,
                self._anti_hebb_coeff,
                x, input_winner_ids,
                input_prewinner_ids,
                input_winning_stat,
                output,
                output_winner_ids,
                output_prewinner_ids,
                output_winning_stat,
                self._descendant_andn_layer._spiking_inhibition,
                w_grad_hebb
            )

            with torch.no_grad():
                m = w_grad_hebb.abs().max()
                if m < 1e-16:
                    m = 1e-16
                w_grad_hebb /= m

            if self._last_w_grad is None:
                # bh_ratio is 1.0
                self._last_w_grad = w_grad_hebb
            else:
                self._last_w_grad.add_(w_grad_hebb, alpha=bh_ratio)

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
        self._output_neuron_ids = self._output_neuron_ids.to(device=self.device)
        self._empty_int_tensor = self._empty_int_tensor.to(device=self.device)
        self._empty_float_tensor = self._empty_float_tensor.to(device=self.device)

        self._andn_dm.to_device(device_index)
        return self

    def train(self, mode=True):
        super().train(mode)
        self._hebbian_history = []
        return self

    def eval(self):
        super().eval()
        self._hebbian_history = []
        return self

    def _push_hebbian_context(self, ctx):
        if self.training and self._n_hebbian_ancestors > 0:
            self._hebbian_history.append(ctx)

    def _pop_hebbian_context(self):
        return self._hebbian_history.pop()

    class ANDNForwardFN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            x, w, andn_layer = args
            ctx.andn_layer = andn_layer
            output, input_winner_ids, input_prewinner_ids, input_winning_stat = andn_layer.forward_step(x)
            ctx.save_for_backward(x, input_winner_ids, input_prewinner_ids, input_winning_stat)
            andn_layer._push_hebbian_context((x, input_winner_ids, input_prewinner_ids, input_winning_stat,))
            return output

        @staticmethod
        def backward(ctx, *grad_outputs):
            (grad_output,) = grad_outputs
            (x, input_winner_ids, input_prewinner_ids, input_winning_stat) = ctx.saved_tensors
            bh_ratio = ctx.andn_layer._backprop_hebb_ratio_on_torch_backward

            if ctx.andn_layer._descendant_andn_layer is not None:
                (
                    output,
                    output_winner_ids,
                    output_prewinner_ids,
                    output_winning_stat
                ) = ctx.andn_layer._descendant_andn_layer._pop_hebbian_context()
            else:
                output = None
                output_winner_ids = None
                output_prewinner_ids = None
                output_winning_stat = None

            x_grad, w_grad = ctx.andn_layer.backward_step(
                x, grad_output,
                input_winner_ids, input_prewinner_ids, input_winning_stat,
                output, output_winner_ids, output_prewinner_ids, output_winning_stat,
                bh_ratio=bh_ratio
            )

            return x_grad, w_grad, None

    def forward(self, x):
        return ANDNLayer.ANDNForwardFN.apply(x, self._weights, self)

    def _count_synapses(self, neuron_ids: torch.Tensor, forward_or_backward: True):
        return self._andn_dm.count_synapses(neuron_ids, forward_or_backward)

    def _export_synapses(
        self, neuron_ids: torch.Tensor,
        source_ids: torch.Tensor,
        weights: torch.Tensor,
        target_ids: torch.Tensor,
        forward_or_backward: True,
        synapse_metas: torch.Tensor = None
    ):
        self._andn_dm.export_synapses(
            self._weights,
            neuron_ids,
            source_ids,
            weights,
            target_ids,
            forward_or_backward,
            synapse_metas
        )


class InhibitionLayer(nn.Module):
    def __init__(
        self, n_inputs,
        n_detectors, max_inputs_per_detector,
        spiking_inhibiton=True,
        backprop_hebb_ratio_on_torch_backward=1.0
    ):
        super().__init__()
        self._empty_int_tensor = torch.tensor([], dtype=torch.int32)
        self._empty_float_tensor = torch.tensor([], dtype=torch.float32)

        self.device = torch.device("cpu")
        self._n_inputs = n_inputs
        self._backprop_hebb_ratio_on_torch_backward = backprop_hebb_ratio_on_torch_backward
        self._spiking_inhibition = spiking_inhibiton

        self._n_detectors = n_detectors
        self._max_inputs_per_detector = max_inputs_per_detector
        self._andn_dm = ANDNDataManagerF(
            n_inputs, 0,
            0, 1, 1,
            spiking_inhibiton
        )
        # TODO support summation_dtype and ANDNDataManagerI (for gradients)

        self._andn_dm.initialize_neurons()
        self._input_neuron_ids = torch.arange(0, n_inputs, dtype=torch.int32, device=self.device)

        self._hebbian_history = []
        self._n_hebbian_ancestors = 0

    def initialize_detectors(self, detectors):
        assert self._n_detectors > 0
        assert detectors.device == self.device
        assert detectors.shape == (self._n_detectors, self._max_inputs_per_detector)
        # detectors tensor must contain input neuron indices for each detector
        # it has shape [n_detectors, max_inputs_per_detector], -1-s are ignored
        assert (detectors != -1).sum(dim=-1).min() >= 2
        self._andn_dm.initialize_detectors(detectors.flatten().contiguous(), self._max_inputs_per_detector)

    def get_input_neuron_ids(self):
        return self._input_neuron_ids

    def n_input_neurons(self):
        return self._n_inputs

    def n_detectors(self):
        return self._n_detectors

    def output_shape(self):
        return (self._n_inputs,)

    def __repr__(self):
        return f'InhibitionLayer({self.n_input_neurons()} inputs, {self.n_detectors()} detectors, {self._andn_dm})'

    def get_smallest_distinguishable_fraction(self) -> float:
        return self._andn_dm.get_smallest_distinguishable_fraction()

    def get_epsilon(self) -> float:
        return self._andn_dm.get_epsilon()

    def get_summation_type(self) -> torch.dtype:
        t = self._andn_dm.get_summations_data_type()
        if t == 'int32':
            return torch.int32
        elif t == 'float32':
            return torch.float32
        else:
            raise RuntimeError('Unsupported summations type: ' + t)

    def get_memory_stats(self) -> str:
        return self._andn_dm.get_memory_stats()

    def get_profiling_stats(self) -> str:
        return self._andn_dm.get_profiling_stats()

    def reset_profiler(self):
        self._andn_dm.reset_profiler()

    def forward_step(self, x):
        assert x.device == self.device
        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        x = x.flatten().contiguous()

        input_winner_ids = torch.zeros([batch_size * self._n_detectors], dtype=torch.int32, device=self.device)
        input_prewinner_ids = torch.zeros([batch_size * self._n_detectors], dtype=torch.int32, device=self.device)
        input_winning_stat = torch.zeros([batch_size * self._n_inputs], dtype=torch.int32, device=self.device)

        self._andn_dm.forward(
            self._empty_float_tensor,
            batch_size, x,
            self._empty_float_tensor,
            input_winner_ids,
            input_prewinner_ids,
            input_winning_stat
        )

        output = (input_winning_stat == 0).to(dtype=torch.float32, device=x.device)
        if not self._spiking_inhibition:
            with torch.no_grad():
                output *= x
        return (
            output.reshape(source_x_shape),
            input_winner_ids.reshape(batch_size, self._n_detectors),
            input_prewinner_ids.reshape(batch_size, self._n_detectors),
            input_winning_stat.reshape(batch_size, self._n_inputs)
        )

    def backward_step(
        self, x, grad_output,
        input_winner_ids,
        input_prewinner_ids,
        input_winning_stat,
        bh_ratio=1.0
    ):
        assert x.device == self.device
        source_x_shape = x.shape
        batch_size = source_x_shape[0]
        x = x.flatten().contiguous()
        x_grad = torch.zeros_like(x)

        if bh_ratio < 1.0:
            assert grad_output.device == self.device
            assert grad_output.shape == (batch_size, self._n_inputs)
            assert input_winner_ids.device == self.device
            assert input_winner_ids.shape == (batch_size, self._n_detectors)
            assert input_prewinner_ids.device == self.device
            assert input_prewinner_ids.shape == (batch_size, self._n_detectors)
            assert input_winning_stat.device == self.device
            assert input_winning_stat.shape == (batch_size, self._n_inputs)
            grad_output = grad_output.flatten().contiguous()
            input_winner_ids = input_winner_ids.flatten().contiguous()
            input_prewinner_ids = input_prewinner_ids.flatten().contiguous()
            input_winning_stat = input_winning_stat.flatten().contiguous()

            self._andn_dm.backward(
                self._empty_float_tensor,
                batch_size,
                bh_ratio,
                0.0,
                grad_output,
                x, input_winner_ids,
                input_prewinner_ids,
                input_winning_stat,
                self._empty_float_tensor,
                self._empty_int_tensor,
                self._empty_int_tensor,
                self._empty_int_tensor,
                x_grad, self._empty_float_tensor
            )
        return x_grad.reshape(source_x_shape)

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
        self._empty_int_tensor = self._empty_int_tensor.to(device=self.device)
        self._empty_float_tensor = self._empty_float_tensor.to(device=self.device)

        self._andn_dm.to_device(device_index)
        return self

    def train(self, mode=True):
        super().train(mode)
        self._hebbian_history = []
        return self

    def eval(self):
        super().eval()
        self._hebbian_history = []
        return self

    def _push_hebbian_context(self, ctx):
        if self.training and self._n_hebbian_ancestors > 0:
            self._hebbian_history.append(ctx)

    def _pop_hebbian_context(self):
        return self._hebbian_history.pop()

    class InhbitionForwardFN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            x, inhibition_layer = args
            ctx.inhibition_layer = inhibition_layer
            output, input_winner_ids, input_prewinner_ids, input_winning_stat = inhibition_layer.forward_step(x)
            ctx.save_for_backward(x, input_winner_ids, input_prewinner_ids, input_winning_stat)
            inhibition_layer._push_hebbian_context((x, input_winner_ids, input_prewinner_ids, input_winning_stat,))
            return output

        @staticmethod
        def backward(ctx, *grad_outputs):
            (grad_output,) = grad_outputs
            (x, input_winner_ids, input_prewinner_ids, input_winning_stat) = ctx.saved_tensors
            bh_ratio = ctx.inhibition_layer._backprop_hebb_ratio_on_torch_backward

            x_grad = ctx.inhibition_layer.backward_step(
                x, grad_output,
                input_winner_ids, input_prewinner_ids, input_winning_stat,
                bh_ratio
            )

            return x_grad, None

    def forward(self, x):
        return InhibitionLayer.InhbitionForwardFN.apply(x, self)


class Conv2DANDNLayer(ANDNLayer):
    def __init__(
        self, input_shape,
        inhibition_grid_shape,
        receptive_field_shape,
        receptive_field_stride_shape,
        output_kernel_shape,
        n_input_channels=None,
        synapse_meta=SynapseMeta(),
        backprop_hebb_ratio_on_torch_backward=1.0,
        relu_output=False,
        anti_hebb_coeff=0.0,
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _forward_group_size: int = 64,
        _backward_group_size: int = 64,
        random_seed=None,
        device=None
    ):
        c_helper = Conv2DSynapseGrowthHelper(
            input_shape[0], input_shape[1],
            receptive_field_shape[0], receptive_field_shape[1],
            receptive_field_stride_shape[0], receptive_field_stride_shape[1],
            output_kernel_shape[0], output_kernel_shape[1],
            n_input_channels=n_input_channels
        )
        n_inputs = input_shape[0] * input_shape[1] * (1 if n_input_channels is None else n_input_channels)
        n_outputs = c_helper.out_h * c_helper.out_w
        if inhibition_grid_shape is not None:
            assert n_input_channels is None
            i_helper = InhibitionGrid2DHelper(
                input_shape[0], input_shape[1],
                inhibition_grid_shape[0], inhibition_grid_shape[1]
            )
            n_detectors = i_helper.num_win_h * i_helper.num_win_w
            max_inputs_per_detector = inhibition_grid_shape[0] * inhibition_grid_shape[1]
        else:
            i_helper = None
            n_detectors = 0
            max_inputs_per_detector = 0

        super().__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_detectors=n_detectors,
            max_inputs_per_detector=max_inputs_per_detector,
            synapse_metas=[synapse_meta],
            backprop_hebb_ratio_on_torch_backward=backprop_hebb_ratio_on_torch_backward,
            relu_output=relu_output,
            anti_hebb_coeff=anti_hebb_coeff,
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            _initial_synapse_capacity=c_helper.n_connections(),
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size
        )

        if n_detectors > 0:
            self.initialize_detectors(
                i_helper.create_detectors(
                    input_ids=self.get_input_neuron_ids().reshape(input_shape)
                )
            )

        if device is not None:
            self.to(device=device)
        else:
            device = torch.device("cpu")

        self.add_connections(
            chunk_of_connections=c_helper.grow_synapses(
                input_ids=self.get_input_neuron_ids().view(input_shape if n_input_channels is None else (input_shape + (n_input_channels,))) + 1,
                output_ids=self.get_output_neuron_ids().view(c_helper.out_h, c_helper.out_w) + 1,
                device=device,
                seed=random_seed
            ),
            ids_shift=-1,
            random_seed=random_seed
        )

        self.compile_andn()
        self._output_shape = (c_helper.out_h, c_helper.out_w)
        self._receptive_field_shape = receptive_field_shape

    def output_shape(self):
        return self._output_shape

    def receptive_field_shape(self):
        return self._receptive_field_shape

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

        return weights[order].reshape(self._output_shape + self._receptive_field_shape)


class Grid2DInhibitionLayer(InhibitionLayer):
    def __init__(
        self, input_shape,
        inhibition_grid_shape,
        spiking_inhibition=True,
        device=None
    ):
        n_inputs = input_shape[0] * input_shape[1]
        i_helper = InhibitionGrid2DHelper(
            input_shape[0], input_shape[1],
            inhibition_grid_shape[0], inhibition_grid_shape[1]
        )
        n_detectors = i_helper.num_win_h * i_helper.num_win_w
        assert n_detectors > 0
        max_inputs_per_detector = inhibition_grid_shape[0] * inhibition_grid_shape[1]

        super().__init__(
            n_inputs=n_inputs,
            n_detectors=n_detectors,
            max_inputs_per_detector=max_inputs_per_detector,
            spiking_inhibiton=spiking_inhibition
        )

        self.initialize_detectors(
            i_helper.create_detectors(
                input_ids=self.get_input_neuron_ids().reshape(input_shape)
            )
        )

        if device is not None:
            self.to(device=device)

        self._output_shape = input_shape

    def output_shape(self):
        return self._output_shape


class Random2DInhibitionLayer(InhibitionLayer):
    def __init__(
        self, input_shape,
        inhibition_window_shape,
        n_detectors,
        max_inputs_per_detector,
        spiking_inhibition=True,
        device=None,
        seed=None
    ):
        assert n_detectors > 0
        n_inputs = input_shape[0] * input_shape[1]
        i_helper = RandomInhibition2DHelper(
            input_shape[0], input_shape[1],
            inhibition_window_shape[0], inhibition_window_shape[1],
            n_detectors, max_inputs_per_detector
        )

        super().__init__(
            n_inputs=n_inputs,
            n_detectors=n_detectors,
            max_inputs_per_detector=max_inputs_per_detector,
            spiking_inhibiton=spiking_inhibition
        )

        self.initialize_detectors(
            i_helper.create_detectors(
                input_ids=self.get_input_neuron_ids().reshape(input_shape),
                seed=seed
            )
        )

        if device is not None:
            self.to(device=device)

        self._output_shape = input_shape

    def output_shape(self):
        return self._output_shape
