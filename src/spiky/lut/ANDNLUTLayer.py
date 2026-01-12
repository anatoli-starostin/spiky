import torch
import torch.nn as nn
from typing import List, Dict, Tuple, AnyStr
from dataclasses import dataclass
from enum import Enum

from spiky.lut.LUTLayer import LUTLayerBasic, SynapseMeta, GradientPolicy, LUTSharedContext
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper
from spiky.util.chunk_of_connections import ChunkOfConnections, create_identity_mapping
from spiky.andn.ANDNLayer import SynapseMeta as ANDN_sm
from spiky.andn.ANDNLayer import Conv2DANDNLayer, Grid2DInhibitionLayer


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
