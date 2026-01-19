import torch
import torch.nn as nn
from typing import List, Dict, Tuple, AnyStr
from dataclasses import dataclass
from enum import Enum

from spiky.lut.LUTLayer import LUTLayerBasic, SynapseMeta, GradientPolicy, LUTSharedContext
from spiky.util.synapse_growth import (
    Conv2DSynapseGrowthHelper, RandomRectanglesSynapseGrowthHelper, GivenRectanglesSynapseGrowthHelper
)
from spiky.util.chunk_of_connections import (
    ChunkOfConnections, create_identity_mapping, repeat_connections_incrementing_source
)
from spiky.andn.ANDNLayer import SynapseMeta as ANDN_sm
from spiky.andn.ANDNLayer import Conv2DANDNLayer, Random2DInhibitionLayer, ANDNLayer


class ANDNLUTLayer(LUTLayerBasic):
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
        synapse_meta=ANDN_sm(),
        concatenation_product=True,
        sliced_product_mode=False,
        positional_dim=None,
        use_sinusoidal_pe=False,
        unified_pe=False,
        backprop_hebb_ratio_on_torch_backward=0.5,
        anti_hebb_coeff=0.0,
        weights_gradient_policy: GradientPolicy = None,
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
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

        self._input_shape = input_shape
        self._detectors_shape = detectors_shape

        super().__init__(
            n_inputs=n_inputs, n_outputs=n_detectors * n_lut_channels, n_detectors=n_detectors,
            n_anchors_per_detector=n_anchors_per_detector, is_fully_connected=False,
            sequence_length=sequence_length, synapse_metas=[SynapseMeta(learning_rate=0.0, initial_weight=1.0)],
            concatenation_product=concatenation_product, sliced_product_mode=sliced_product_mode,
            positional_dim=positional_dim, use_sinusoidal_pe=use_sinusoidal_pe, unified_pe=unified_pe,
            weights_gradient_policy=weights_gradient_policy,
            shared_context=shared_context,
            summation_dtype=summation_dtype, _int_rescaler=_int_rescaler,
            _initial_synapse_capacity=n_detectors * n_lut_channels,
            _forward_group_size=1,
            _backward_group_size=1,
            random_seed=random_seed
        )

        if device is not None:
            self.to(device=device)
        else:
            device = torch.device("cpu")

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

        self.add_lookup_connections(
            chunk_of_connections=create_identity_mapping(
                n_detectors * n_lut_channels,
                delta=n_detectors * n_lut_channels,
                device=device
            ),
            ids_shift=-1,
            random_seed=random_seed
        )

        self.compile_lut()

        self._andn_layer = Conv2DANDNLayer(
            lut_shape, None,
            lut_receptive_field_shape,
            lut_receptive_field_stride_shape,
            output_kernel_shape,
            n_input_channels=n_lut_channels,
            synapse_meta=synapse_meta,
            backprop_hebb_ratio_on_torch_backward=backprop_hebb_ratio_on_torch_backward,
            relu_output=False,
            anti_hebb_coeff=anti_hebb_coeff,
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size,
            random_seed=random_seed,
            device=device
        )
        self._output_shape = (lut_shape[0], lut_shape[1] * n_lut_channels)
        self._inhibition_layer = Grid2DInhibitionLayer(
            self._andn_layer.output_shape(),
            output_kernel_shape,
            spiking_inhibition=False,
            device=device
        )
        self._andn_layer.set_descendant_andn_layer(self._inhibition_layer)

    def forward(self, x):
        x = super().forward(x)
        x = self._andn_layer(x)
        return self._inhibition_layer(x)

    def input_shape(self):
        return self._input_shape

    def output_shape(self):
        return self._output_shape

    def detectors_shape(self):
        return self._detectors_shape

    def __repr__(self):
        return f'ANDNLUTLayer(input_shape={self.input_shape()}, output_shape={self.output_shape()}, detectors_shape={self.detectors_shape()}, n_anchors_per_detector={self.n_anchors_per_detector()})'


class _AuxANDNLayer(ANDNLayer):
    def __init__(
        self, n_inputs, output_shape,
        group_centers,
        projection_shape,
        n_projections_per_detector,
        n_detectors_in_group,
        n_lut_channels,
        synapse_meta,
        backprop_hebb_ratio_on_torch_backward=1.0,
        relu_output=False,
        anti_hebb_coeff=0.0,
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _forward_group_size: int = 64,
        _backward_group_size: int = 64,
        _max_groups_in_growth_buffer=2 ** 20,
        random_seed=None,
        device=None
    ):
        super().__init__(
            n_inputs=n_inputs,
            n_outputs=output_shape[0] * output_shape[1],
            n_detectors=0,
            max_inputs_per_detector=0,
            synapse_metas=[synapse_meta],
            backprop_hebb_ratio_on_torch_backward=backprop_hebb_ratio_on_torch_backward,
            relu_output=relu_output,
            anti_hebb_coeff=anti_hebb_coeff,
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            _initial_synapse_capacity=None,
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size
        )

        if device is not None:
            self.to(device=device)

        c_helper = GivenRectanglesSynapseGrowthHelper(
            group_centers[::n_detectors_in_group],
            projection_shape[0], projection_shape[1],
            output_shape[0], output_shape[1],
            max_synapses_per_input=n_projections_per_detector
        )

        connections = c_helper.grow_synapses(
            input_ids=self.get_input_neuron_ids()[::n_detectors_in_group * n_lut_channels] + 1,
            output_ids=self.get_output_neuron_ids().reshape(output_shape) + 1,
            max_groups_in_buffer=_max_groups_in_growth_buffer,
            device=device,
            seed=random_seed
        )

        connections = repeat_connections_incrementing_source(connections, n_lut_channels * n_detectors_in_group)

        self.add_connections(
            chunk_of_connections=connections,
            ids_shift=-1,
            random_seed=random_seed
        )

        self.compile_andn()
        self._output_shape = output_shape

    def output_shape(self):
        return self._output_shape


class ANDNLUTLayerEx(LUTLayerBasic):
    def __init__(
        self, input_shape, output_shape,
        n_anchors_per_detector,
        n_detector_groups,
        n_detectors_in_group,
        receptive_shape,
        projection_shape,
        n_projections_per_detector,
        inhibition_window_shape,
        n_inhibitors,
        n_neurons_per_inhibitor,
        synapse_meta=ANDN_sm(),
        backprop_hebb_ratio_on_torch_backward=0.5,
        anti_hebb_coeff=0.0,
        relu_before_inhibition=True,
        residual=False,
        weights_gradient_policy: GradientPolicy = None,
        shared_context: LUTSharedContext = None,
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _forward_group_size=32,
        _backward_group_size=32,
        _max_groups_in_growth_buffer=2 ** 20,
        random_seed=None,
        device=None
    ):
        assert n_detector_groups >= 1
        assert n_detectors_in_group >= 1
        c_helper = RandomRectanglesSynapseGrowthHelper(
            input_shape[0], input_shape[1],
            receptive_shape[0], receptive_shape[1],
            input_shape[0], input_shape[1],
            n_outputs=n_detector_groups,
            n_out_channels=n_detectors_in_group
        )

        n_inputs = input_shape[0] * input_shape[1]

        self._input_shape = input_shape
        self._residual = residual
        if residual:
            assert input_shape == output_shape
        n_lut_channels = LUTLayerBasic.n_lut_channels(n_anchors_per_detector, 1)

        super().__init__(
            n_inputs=n_inputs, n_outputs=n_detector_groups * n_detectors_in_group * n_lut_channels,
            n_detectors=n_detector_groups * n_detectors_in_group,
            n_anchors_per_detector=n_anchors_per_detector, is_fully_connected=False,
            sequence_length=1, synapse_metas=[SynapseMeta(learning_rate=0.0, initial_weight=1.0)],
            weights_gradient_policy=weights_gradient_policy,
            shared_context=shared_context,
            summation_dtype=summation_dtype, _int_rescaler=_int_rescaler,
            _initial_synapse_capacity=n_detector_groups * n_detectors_in_group * n_lut_channels,
            _forward_group_size=1,
            _backward_group_size=1,
            random_seed=random_seed
        )

        if device is not None:
            self.to(device=device)
        else:
            device = torch.device("cpu")

        connections, group_centers = c_helper.grow_synapses(
            input_ids=self.get_input_neuron_ids().view(input_shape) + 1,
            output_ids=self.get_detector_neuron_ids() + 1,
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

        self.add_lookup_connections(
            chunk_of_connections=create_identity_mapping(
                n_detector_groups * n_detectors_in_group * n_lut_channels,
                delta=n_detector_groups * n_detectors_in_group * n_lut_channels,
                device=device
            ),
            ids_shift=-1,
            random_seed=random_seed
        )

        self.compile_lut()

        group_centers[:, 0] *= output_shape[0] / input_shape[0]
        group_centers[:, 1] *= output_shape[1] / input_shape[1]

        self._andn_layer = _AuxANDNLayer(
            n_inputs=n_detector_groups * n_detectors_in_group * n_lut_channels,
            output_shape=output_shape,
            group_centers=group_centers,
            projection_shape=projection_shape,
            n_projections_per_detector=n_projections_per_detector,
            n_detectors_in_group=n_detectors_in_group,
            n_lut_channels=n_lut_channels,
            synapse_meta=synapse_meta,
            backprop_hebb_ratio_on_torch_backward=backprop_hebb_ratio_on_torch_backward,
            relu_output=relu_before_inhibition,
            anti_hebb_coeff=anti_hebb_coeff,
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size,
            _max_groups_in_growth_buffer=_max_groups_in_growth_buffer,
            random_seed=random_seed,
            device=device
        )
        self._output_shape = output_shape
        self._inhibition_layer = Random2DInhibitionLayer(
            self._andn_layer.output_shape(),
            inhibition_window_shape,
            n_inhibitors,
            n_neurons_per_inhibitor,
            spiking_inhibition=False,
            device=device,
            seed=random_seed
        )
        self._andn_layer.set_descendant_andn_layer(self._inhibition_layer)

    def forward(self, x):
        source_x = super().forward(x)
        x = self._andn_layer(x)
        if self._residual:
            x = x + source_x
        return self._inhibition_layer(x)

    def input_shape(self):
        return self._input_shape

    def output_shape(self):
        return self._output_shape

    def detectors_shape(self):
        return self._detectors_shape

    def __repr__(self):
        return f'ANDNLUTLayerEx(input_shape={self.input_shape()}, output_shape={self.output_shape()}, n_detectors={self._n_detectors}, n_anchors_per_detector={self.n_anchors_per_detector()})'
