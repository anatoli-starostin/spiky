import torch
import torch.nn as nn

from spiky.andn.ANDNLayer import (
    Conv2DANDNLayer, SynapseMeta, ANDNLayer, Grid2DInhibitionLayer, SynapseMeta
)
from spiky.util.synapse_growth import SynapseGrowthEngine
from spiky.util.chunk_of_connections import ChunkOfConnections


class Conv2DInverseANDNLayer(ANDNLayer):
    def __init__(
        self, straight_andn_layer, output_shape,
        synapse_meta=SynapseMeta(),
        backprop_hebb_ratio_on_torch_backward=1.0,
        anti_hebb_coeff=0.0,
        summation_dtype=torch.float32,
        _int_rescaler=0.001,
        _forward_group_size: int = 64,
        _backward_group_size: int = 64,
        random_seed=None,
        device=None
    ):
        input_shape = straight_andn_layer.output_shape()
        n_inputs = input_shape[0] * input_shape[1]
        n_outputs = output_shape[0] * output_shape[1]
        print(input_shape)
        print(output_shape)
        n_synapses = straight_andn_layer._count_synapses(straight_andn_layer.get_input_neuron_ids(), True)
        print(n_synapses)

        super().__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_detectors=0,
            max_inputs_per_detector=0,
            synapse_metas=[synapse_meta],
            backprop_hebb_ratio_on_torch_backward=backprop_hebb_ratio_on_torch_backward,
            anti_hebb_coeff=anti_hebb_coeff,
            summation_dtype=summation_dtype,
            _int_rescaler=_int_rescaler,
            _initial_synapse_capacity=n_synapses,
            _forward_group_size=_forward_group_size,
            _backward_group_size=_backward_group_size
        )

        if device is not None:
            self.to(device=device)
        else:
            device = torch.device("cpu")

        synapses_export = {
            'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
            'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
            'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
        }

        straight_andn_layer._export_synapses(
            straight_andn_layer.get_input_neuron_ids(),
            synapses_export['source_ids'],
            synapses_export['weights'],
            synapses_export['target_ids'],
            forward_or_backward=True
        )

        explicit_triples = torch.cat(
            [
                torch.zeros([n_synapses, 1], dtype=torch.int32, device=device),
                synapses_export['target_ids'].unsqueeze(1) - n_outputs + 1,
                synapses_export['source_ids'].unsqueeze(1) + n_inputs + 1
            ],
            dim=-1
        )

        growth_engine = SynapseGrowthEngine(
            device=device, synapse_group_size=64,
            max_groups_in_buffer=2 ** 10
        )
        growth_engine._max_neuron_id = explicit_triples[:, 1:].max().item()

        chunk_of_connections = growth_engine._grow_explicit(explicit_triples, random_seed, False)
        chunk_of_connections._connections = chunk_of_connections._connections.to(device=device)

        self.add_connections(
            chunk_of_connections=chunk_of_connections,
            ids_shift=-1,
            random_seed=random_seed
        )

        self.compile_andn()
        self._output_shape = output_shape

    def output_shape(self):
        return self._output_shape

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


synapse_meta = SynapseMeta(
    min_weight=0.0,
    max_weight=1.0,
    initial_weight=1.0,
    initial_noise_level=-1.0
)

backprop_hebb_ratio = 0.0

cl_dim = 6
p_dim = 5

andn_shapes = [
    {
        'inhibition': None,
        'straight': {
            'receptive_field': (p_dim, p_dim),
            'receptive_field_stride': (1, 1),
            'output_kernel': (cl_dim, cl_dim)
        },
        'inverse': {
            'receptive_field': (p_dim * cl_dim, p_dim * cl_dim),
            'receptive_field_stride': (cl_dim, cl_dim),
            'output_kernel': (1, 1)
        }
    },
    {
        'inhibition': (cl_dim, cl_dim),
        'straight': {
            'receptive_field': (p_dim * cl_dim, p_dim * cl_dim),
            'receptive_field_stride': (cl_dim, cl_dim),
            'output_kernel': (cl_dim, cl_dim)
        },
        'inverse': {
            'receptive_field': (p_dim * cl_dim, p_dim * cl_dim),
            'receptive_field_stride': (cl_dim, cl_dim),
            'output_kernel': (cl_dim, cl_dim)
        }
    },
    {
        'inhibition': (cl_dim, cl_dim),
        'straight': {
            'receptive_field': (p_dim * cl_dim, p_dim * cl_dim),
            'receptive_field_stride': (cl_dim, cl_dim),
            'output_kernel': (cl_dim, cl_dim)
        },
        'inverse': {
            'receptive_field': (p_dim * cl_dim, p_dim * cl_dim),
            'receptive_field_stride': (1, 1),
            'output_kernel': (1, 1)
        }
    },
]
last_inhibition = (cl_dim, cl_dim)


class ANDNAutoencoder(nn.Module):
    def __init__(self, input_shape, summation_dtype, device, random_seed):
        super().__init__()
        self.straight_layers = []
        self.inhibition_layers = []
        self.inverse_layers = []

        prev_output_shape = input_shape

        for i, sh in enumerate(andn_shapes):
            if sh['inhibition'] is not None:
                inhibition_layer = Grid2DInhibitionLayer(
                    prev_output_shape,
                    sh['inhibition'],
                    device=device
                )
            else:
                inhibition_layer = None
            self.inhibition_layers.append(inhibition_layer)
            print(sh)
            print(prev_output_shape)
            straight_layer = Conv2DANDNLayer(
                input_shape=prev_output_shape,
                inhibition_grid_shape=None,
                receptive_field_shape=sh['straight']['receptive_field'],
                receptive_field_stride_shape=sh['straight']['receptive_field_stride'],
                output_kernel_shape=sh['straight']['output_kernel'],
                backprop_hebb_ratio_on_torch_backward=backprop_hebb_ratio,
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                random_seed=random_seed,
                device=device
            )
            print(i)
            print(straight_layer)
            self.straight_layers.append(straight_layer)
            inverse_layer = Conv2DInverseANDNLayer(
                straight_layer, prev_output_shape,
                backprop_hebb_ratio_on_torch_backward=backprop_hebb_ratio,
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                random_seed=random_seed,
                device=device
            )
            prev_output_shape = straight_layer.output_shape()
            self.inverse_layers.append(inverse_layer)

        self.inhibition_layers.append(
            Grid2DInhibitionLayer(
                prev_output_shape,
                last_inhibition,
                device=device
            )
        )
        self.all_layers = nn.ModuleList(
            [m for m in self.straight_layers + self.inhibition_layers + self.inverse_layers if m is not None]
        )

    def forward(self, x):
        y = x
        i_x = None
        prev_s_l = None
        internal_loss = torch.zeros([1], device=x.device)
        for s_l, inh_l, inv_l in zip(self.straight_layers, self.inhibition_layers[:-1], self.inverse_layers):
            if self.training and prev_s_l is not None:
                i_x = i_x.detach()
                y = prev_s_l(i_x)
                y = inh_l(y)
                x_bar = inv_l(y)
                internal_loss += F.mse_loss(i_x, x_bar, reduction="sum")

            if inh_l is not None:
                i_x = inh_l(y)
            else:
                i_x = y
            y = s_l(i_x)
            prev_s_l = s_l

        return y, internal_loss

    def decode(self, y):
        for inh_l, inv_l in reversed(zip(self.inhibition_layers, self.inverse_layers[1:])):
            if inh_l is not None:
                y = inh_l(y)
            y = inv_l(y)
        return y


def test_andn_autoencoder(
    device, summation_dtype, seed=None
):
    success = _test_andn_layer_forward(
        (28, 28),
        summation_dtype,
        device,
        seed
    )
    return success


def _test_andn_layer_forward(
    input_shape, summation_dtype,
    device, seed
):
    andn_net = ANDNAutoencoder(
        input_shape, summation_dtype,
        device, seed
    )
    return True


def main():
    print("=" * 60)
    print("ANDNLayer AUTOENCODER TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_andn_autoencoder(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
