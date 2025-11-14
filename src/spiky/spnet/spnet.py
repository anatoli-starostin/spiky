import torch
import math
from enum import Enum
from dataclasses import dataclass, replace
from typing import List, Dict, AnyStr

from spiky_cuda import SPNetDataManagerF, SPNetDataManagerI
from spiky.util.synapse_growth import ChunkOfConnections


@dataclass(frozen=True, order=True)
class NeuronMeta:
    # Izhikevitch model:
    #   v' = cf_2 * v**2 + cf_1 * v + cf_0 - u - I
    #   u' = a(b * v - u)
    #   if v >= spike_threshold:
    #     v = c
    #     u = u + d
    neuron_type: int
    cf_2: float = 0.04
    cf_1: float = 5.0
    cf_0: float = 140.0
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    spike_threshold: float = 30.0
    stdp_decay: float = 0.95
    ltp_max: float = 1.0
    ltd_max: float = 1.2

    def __post_init__(self):
        assert 0.0 < self.stdp_decay < 1.0
        assert 0.0 <= self.ltp_max
        assert 0.0 <= self.ltd_max


@dataclass(frozen=True, order=True)
class SynapseMeta:
    # Update rule
    #   (s - some synaptic weight, sd - it's derivative):
    #   s += weight_scaling_cf + sd * learning_rate
    #   sd *= weight_decay
    #   sd = sd.clip(min_weight, max_weight)

    learning_rate: float = 0.05
    min_delay: int = 0
    max_delay: int = 0
    min_weight: float = 0.0
    max_weight: float = 10.0
    initial_noise_level: float = 0.0
    initial_weight: float = 0.0
    weight_decay: float = 0.9
    weight_scaling_cf: float = 0.01
    _forward_group_size: int = 64
    _backward_group_size: int = 64

    def __post_init__(self):
        assert 0.0 <= self.learning_rate < 1.0
        assert 0 <= self.min_delay <= 255
        assert 0 <= self.max_delay <= 255
        assert self.min_delay <= self.max_delay
        assert self.min_weight <= self.initial_weight <= self.max_weight
        assert 0.0 < self.weight_decay <= 1.0
        assert 0.0 <= self.weight_scaling_cf <= 0.1
        assert self._forward_group_size > 0
        assert self._backward_group_size > 0


class NeuronDataType(Enum):
    Spike = 0
    Voltage = 1


class SpikingNet(object):
    def __init__(
        self, synapse_metas: List[SynapseMeta], neuron_metas: List[NeuronMeta], neuron_counts: List[int],
        initial_synapse_capacity=1024, summation_dtype: torch.dtype = torch.float32,
        stdp_threshold: float = 0.00001
    ):
        assert len(synapse_metas) > 0
        assert len(neuron_metas) > 0
        assert len(neuron_metas) == len(neuron_counts)
        assert all(n > 0 for n in neuron_counts)

        if summation_dtype == torch.float32:
            self._neuron_data_manager = SPNetDataManagerF(initial_synapse_capacity, stdp_threshold)
        else:
            self._neuron_data_manager = SPNetDataManagerI(initial_synapse_capacity, stdp_threshold)

        for i, sm in enumerate(synapse_metas):
            m_id = self._neuron_data_manager.register_synapse_meta(
                learning_rate=sm.learning_rate,
                min_delay=sm.min_delay,
                max_delay=sm.max_delay,
                min_synaptic_weight=sm.min_weight,
                max_synaptic_weight=sm.max_weight,
                initial_noise_level=sm.initial_noise_level,
                initial_weight=sm.initial_weight,
                weight_decay=sm.weight_decay,
                weight_scaling_cf=sm.weight_scaling_cf,
                _forward_group_size=sm._forward_group_size,
                _backward_group_size=sm._backward_group_size
            )
            assert m_id == i

        for i, nm in enumerate(neuron_metas):
            m_id = self._neuron_data_manager.register_neuron_meta(
                neuron_type=nm.neuron_type,
                cf_2=nm.cf_2,
                cf_1=nm.cf_1,
                cf_0=nm.cf_0,
                a=nm.a,
                b=nm.b,
                c=nm.c,
                d=nm.d,
                spike_threshold=nm.spike_threshold,
                stdp_decay=nm.stdp_decay,
                ltp_max=nm.ltp_max,
                ltd_max=nm.ltd_max
            )
            assert m_id == i

        neuron_counts4 = [((c + 3) // 4) * 4 for c in neuron_counts]

        ids_shift = self._neuron_data_manager.initialize_neurons(torch.tensor(neuron_counts4, dtype=torch.int32))
        self._neuron_ids_by_meta_ids = []
        for n, n4 in zip(neuron_counts, neuron_counts4):
            neuron_ids = torch.arange(ids_shift, ids_shift + n, dtype=torch.int32)
            self._neuron_ids_by_meta_ids.append(neuron_ids)
            ids_shift += n4

        assert self.n_neurons() > 0

    def get_neuron_ids_by_meta(self, neuron_meta_index):
        return self._neuron_ids_by_meta_ids[neuron_meta_index]

    def get_all_neuron_ids(self):
        return torch.cat(self._neuron_ids_by_meta_ids)

    def n_neurons(self):
        return sum([n_ids.numel() for n_ids in self._neuron_ids_by_meta_ids])

    def n_synapses(self):
        return self._neuron_data_manager.get_total_number_of_synapses()

    def max_delay(self):
        return self._neuron_data_manager.get_max_delay()

    def __repr__(self):
        return f'SpikingNet({self.n_neurons()} neurons, {self.n_synapses()} synapses, {self._neuron_data_manager})'

    def add_connections(
        self, chunk_of_connections: ChunkOfConnections,
        random_seed: int = None
    ):
        self._neuron_data_manager.add_connections(
            chunk_of_connections.get_connections(),
            chunk_of_connections.get_single_group_size(),
            random_seed
        )

    def compile(self, shuffle_synapses_random_seed: int = None, _only_trainable_backwards=True):
        self._neuron_data_manager.compile(_only_trainable_backwards, shuffle_synapses_random_seed)

    def get_smallest_distinguishable_fraction(self) -> float:
        return self._neuron_data_manager.get_smallest_distinguishable_fraction()

    def get_epsilon(self) -> float:
        return self._neuron_data_manager.get_epsilon()

    def get_summation_type(self) -> torch.dtype:
        t = self._neuron_data_manager.get_summations_data_type()
        if t == 'int32':
            return torch.int32
        elif t == 'float32':
            return torch.float32
        else:
            raise RuntimeError('Unsupported summations type: ' + t)

    def get_memory_stats(self) -> str:
        return self._neuron_data_manager.get_memory_stats()

    def get_profiling_stats(self) -> str:
        return self._neuron_data_manager.get_profiling_stats()

    def reset_profiler(self):
        self._neuron_data_manager.reset_profiler()

    def to_device(self, device):
        if not isinstance(device, int):
            device = str(device)
            if device.startswith('cuda'):
                s = device.split(':')
                device = int(s[1]) if len(s) == 2 else 0
            elif device == 'cpu':
                device = -1

        if device == -1:
            str_device = 'cpu'
        else:
            str_device = f'cuda:{device}'
            # to ensure the creation of torch cuda context
            torch.tensor([1.0], device=str_device)

        self._neuron_data_manager.to_device(device)
        for i in range(len(self._neuron_ids_by_meta_ids)):
            neuron_ids = self._neuron_ids_by_meta_ids[i]
            self._neuron_ids_by_meta_ids[i] = neuron_ids.to(device=str_device)

    def get_device(self):
        return self._neuron_ids_by_meta_ids[0].device

    def process_ticks(
        self, n_ticks_to_process: int, batch_size: int, n_input_ticks,
        input_values: torch.Tensor, do_train: bool,
        input_neuron_ids: torch.Tensor = None, sparse_input=None,
        do_reset_context: bool = False, do_apply_deltas=True, do_record_voltage: bool = False,
        _stdp_period=None
    ):
        if sparse_input is not None:
            if input_neuron_ids is not None:
                assert input_values.shape[-1] == sparse_input.shape[-1]
                assert sparse_input.max().item() < n_input_ticks
            else:
                assert input_values.shape[-1] == sparse_input.shape[-1]
                assert sparse_input.max().item() < self._neuron_data_manager.get_total_number_of_neurons()
        else:
            assert input_values.shape[-1] == n_input_ticks

        if _stdp_period is None:
            if self.get_device().type == "cpu":
                _stdp_period = 1
            else:
                _stdp_period = 32

        return self._neuron_data_manager.process_ticks(
            n_ticks_to_process, batch_size, n_input_ticks,
            input_values.flatten().contiguous(),
            do_train, do_reset_context, do_apply_deltas, do_record_voltage, _stdp_period,
            input_neuron_ids.flatten().contiguous() if input_neuron_ids is not None else None,
            sparse_input.flatten().contiguous() if sparse_input is not None else None
        )

    # TODO support sparse output (neuron, <tick_i, val_i>...)
    def export_neuron_data(
        self, neuron_ids: torch.Tensor,
        batch_size: int, data_type: object,
        first_tick: int, last_tick: int
    ) -> torch.Tensor:
        res = torch.zeros(
            [batch_size * (last_tick - first_tick + 1) * neuron_ids.numel()], dtype=torch.float32,
            device=self.get_device()
        )
        if isinstance(data_type, NeuronDataType):
            self._neuron_data_manager.export_neuron_state_info(
                res, neuron_ids, batch_size, data_type.value,
                first_tick, last_tick
            )
        else:
            raise RuntimeError('Unsupported data_type: ' + str(data_type))
        return res.view(batch_size, neuron_ids.numel(), last_tick - first_tick + 1)

    def export_input_synaptic_weights(self, neuron_ids: torch.Tensor) -> torch.Tensor:
        assert len(neuron_ids.shape) == 1
        max_input_weights_per_neuron = self._neuron_data_manager.count_max_input_synapses_per_neuron(neuron_ids)
        target_tensor = torch.zeros(
            [max_input_weights_per_neuron * neuron_ids.numel()], dtype=torch.float32,
            device=self.get_device()
        )
        target_tensor[:] = float('nan')
        self._neuron_data_manager.export_input_synaptic_weights(
            target_tensor, neuron_ids
        )

        return target_tensor.reshape(
            neuron_ids.numel(), max_input_weights_per_neuron
        )

    def count_synapses(self, neuron_ids: torch.Tensor, forward_or_backward: True):
        return self._neuron_data_manager.count_synapses(neuron_ids, forward_or_backward)

    def export_synapses(
        self, neuron_ids: torch.Tensor,
        source_ids: torch.Tensor,
        synapse_metas: torch.Tensor,
        weights: torch.Tensor,
        delays: torch.Tensor,
        target_ids: torch.Tensor,
        forward_or_backward: True
    ):
        self._neuron_data_manager.export_synapses(
            neuron_ids,
            source_ids,
            synapse_metas,
            weights,
            delays,
            target_ids,
            forward_or_backward
        )
