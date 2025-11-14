import torch
import torch.nn as nn

from spiky.andn.ANDNLayer import Conv2DANDNLayer, SynapseMeta
from spiky.util.test_utils import lex_idx
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper, InhibitionGrid2DHelper


def create_gt_connections(
    input_shape,
    receptive_field_shape,
    receptive_field_stride_shape,
    output_kernel_shape,
    device
):
    c_helper = Conv2DSynapseGrowthHelper(
        input_shape[0], input_shape[1],
        receptive_field_shape[0], receptive_field_shape[1],
        receptive_field_stride_shape[0], receptive_field_stride_shape[1],
        output_kernel_shape[0], output_kernel_shape[1]
    )

    ground_truth_pairs = set()
    for win_y in range(c_helper.num_win_h):
        for win_x in range(c_helper.num_win_w):
            for j in range(c_helper.rh):
                for i in range(c_helper.rw):
                    source_id = (win_y * c_helper.sh + j) * c_helper.w + win_x * c_helper.sw + i
                    oy = win_y * c_helper.kh
                    ox = win_x * c_helper.kw
                    for by in range(c_helper.kh):
                        for bx in range(c_helper.kw):
                            target_id = c_helper.h * c_helper.w + (oy + by) * c_helper.out_w + ox + bx
                            ground_truth_pairs.add((source_id, target_id,))

    source_ids = torch.zeros([len(ground_truth_pairs)], dtype=torch.int32, device=torch.device('cpu'))
    targets_ids = torch.zeros([len(ground_truth_pairs)], dtype=torch.int32, device=torch.device('cpu'))

    for i, (source_id, target_id) in enumerate(sorted(ground_truth_pairs)):
        source_ids[i] = source_id
        targets_ids[i] = target_id

    return source_ids.to(device), targets_ids.to(device)


def connections_to_matrix(n_inputs, n_outputs, source_ids, targets_ids, weights, device):
    res = torch.zeros(n_inputs, n_outputs, dtype=torch.float32, device=device)
    res[source_ids, targets_ids - n_inputs] = weights
    return res


def gt_detectors(input_shape, inhibition_shape, device):
    i_helper = InhibitionGrid2DHelper(
        input_shape[0], input_shape[1],
        inhibition_shape[0], inhibition_shape[1]
    )
    return i_helper.create_detectors(
        torch.arange(
            0, input_shape[0] * input_shape[1],
            dtype=torch.int32, device=device
        ).reshape(input_shape)
    )


def test_andn_layer_forward(
    device, summation_dtype, seed=123
):
    success = _test_andn_layer_forward(
        input_shape=(4, 4),
        receptive_field_shape=(2, 2),
        receptive_field_stride_shape=(1, 1),
        output_kernel_shape=(2, 2),
        forward_group_size=3,
        backward_group_size=3,
        device=device,
        summation_dtype=summation_dtype,
        seed=seed
    )
    success = success and _test_andn_layer_forward(
        input_shape=(28, 28),
        receptive_field_shape=(5, 5),
        receptive_field_stride_shape=(1, 1),
        output_kernel_shape=(6, 6),
        forward_group_size=64,
        backward_group_size=64,
        device=device,
        summation_dtype=summation_dtype,
        seed=seed
    )
    return success


def _test_andn_layer_forward(
    input_shape,
    receptive_field_shape,
    receptive_field_stride_shape,
    output_kernel_shape,
    device, summation_dtype,
    forward_group_size, backward_group_size,
    batch_size=16, seed=123
):
    torch.manual_seed(seed)
    final_output_shape = (1, 10)
    synapse_meta = SynapseMeta(
        initial_weight=0.0,
        initial_noise_level=1.0
    )

    class TestNet(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.layer1 = Conv2DANDNLayer(
                input_shape=input_shape,
                inhibition_grid_shape=None,
                receptive_field_shape=receptive_field_shape,
                receptive_field_stride_shape=receptive_field_stride_shape,
                output_kernel_shape=output_kernel_shape,
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                _int_rescaler=1.0,
                _forward_group_size=forward_group_size,
                _backward_group_size=backward_group_size,
                random_seed=seed,
                device=device
            )
            self.layer2 = Conv2DANDNLayer(
                input_shape=self.layer1.output_shape(),
                inhibition_grid_shape=output_kernel_shape,
                receptive_field_shape=self.layer1.output_shape(),
                receptive_field_stride_shape=self.layer1.output_shape(),
                output_kernel_shape=final_output_shape,
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                _int_rescaler=1.0,
                _forward_group_size=forward_group_size,
                _backward_group_size=backward_group_size,
                random_seed=seed, device=device
            )
            self.layer1.set_descendant_andn_layer(self.layer2)

        def forward(self, x):
            return self.layer2(self.layer1(x))

    print(f'Creating TestNet, input_shape {input_shape}...')
    net = TestNet(device)

    layer_1_output_shape = (
        (((input_shape[0] - receptive_field_shape[0]) // receptive_field_stride_shape[0]) + 1) * output_kernel_shape[0],
        (((input_shape[1] - receptive_field_shape[1]) // receptive_field_stride_shape[1]) + 1) * output_kernel_shape[1]
    )

    assert net.layer1.output_shape() == layer_1_output_shape

    print(f'Creating gt_connections, layer 1...')

    layer_1_gt_source_ids, layer_1_gt_target_ids = create_gt_connections(
        input_shape=input_shape,
        receptive_field_shape=receptive_field_shape,
        receptive_field_stride_shape=receptive_field_stride_shape,
        output_kernel_shape=output_kernel_shape,
        device=device
    )

    print(f'Creating gt_connections, layer 2...')

    layer_2_gt_source_ids, layer_2_gt_target_ids = create_gt_connections(
        input_shape=layer_1_output_shape,
        receptive_field_shape=layer_1_output_shape,
        receptive_field_stride_shape=layer_1_output_shape,
        output_kernel_shape=final_output_shape,
        device=device
    )

    print(net.layer1)
    print(net.layer2)

    print(f'Exporting synapses, layer 1...')

    n_synapses = net.layer1._count_synapses(net.layer1.get_input_neuron_ids(), True)
    forward_export_1 = {
        'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
        'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
    }

    net.layer1._export_synapses(
        net.layer1.get_input_neuron_ids(),
        forward_export_1['source_ids'],
        forward_export_1['weights'],
        forward_export_1['target_ids'],
        forward_or_backward=True
    )

    order1 = lex_idx(forward_export_1['source_ids'], forward_export_1['target_ids'])

    if torch.any(forward_export_1['source_ids'][order1] != layer_1_gt_source_ids):
        print(f"❌ wrong source connections at layer 1")
        return False

    if torch.any(forward_export_1['target_ids'][order1] != layer_1_gt_target_ids):
        print(f"❌ wrong target connections at layer 1")
        return False

    print(f'Exporting synapses, layer 2...')

    n_synapses = net.layer2._count_synapses(net.layer2.get_input_neuron_ids(), True)
    forward_export_2 = {
        'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
        'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
    }

    net.layer2._export_synapses(
        net.layer2.get_input_neuron_ids(),
        forward_export_2['source_ids'],
        forward_export_2['weights'],
        forward_export_2['target_ids'],
        forward_or_backward=True
    )

    order2 = lex_idx(forward_export_2['source_ids'], forward_export_2['target_ids'])

    if torch.any(forward_export_2['source_ids'][order2] != layer_2_gt_source_ids):
        print(f"❌ wrong source connections at layer 2")
        return False

    if torch.any(forward_export_2['target_ids'][order2] != layer_2_gt_target_ids):
        print(f"❌ wrong target connections at layer 2")
        return False

    print(f'Creating GtNet, input_shape {input_shape}...')

    class GtNet(nn.Module):
        def __init__(
            self,
            input_shape,
            output_kernel_shape,
            layer_1_output_shape,
            source_ids_1,
            targets_ids_1,
            weights_1,
            final_output_shape,
            source_ids_2,
            targets_ids_2,
            weights_2,
            device
        ):
            super().__init__()
            self.m1 = connections_to_matrix(
                input_shape[0] * input_shape[1],
                layer_1_output_shape[0] * layer_1_output_shape[1],
                source_ids_1, targets_ids_1, weights_1, device
            )
            self.m2 = connections_to_matrix(
                layer_1_output_shape[0] * layer_1_output_shape[1],
                final_output_shape[0] * final_output_shape[1],
                source_ids_2, targets_ids_2, weights_2, device
            )
            self.detectors = gt_detectors(layer_1_output_shape, output_kernel_shape, device)

        def forward(self, x):
            p = x @ self.m1
            y = torch.zeros_like(p)
            buckets = p[self.detectors]
            winners = buckets.argmax(dim=-1)
            y[self.detectors[torch.arange(buckets.size(0), device=p.device), winners]] = 1.0
            return (y @ self.m2).reshape(final_output_shape)

    gt_net = GtNet(
        input_shape,
        output_kernel_shape,
        layer_1_output_shape,
        layer_1_gt_source_ids,
        layer_1_gt_target_ids,
        forward_export_1['weights'][order1],
        final_output_shape,
        layer_2_gt_source_ids,
        layer_2_gt_target_ids,
        forward_export_2['weights'][order2],
        device
    )

    print(f'Calculating ground truth...')

    x = torch.rand([batch_size, input_shape[0] * input_shape[1]], device=device)
    gt_out = []
    for i in range(batch_size):
        gt_out.append(gt_net(x[i]))
    gt_out = torch.stack(gt_out, dim=0)

    print(f'Calculating result...')

    y = net(x.reshape(batch_size, input_shape[0], input_shape[1]))

    if (gt_out - y).abs().max() > 0.001:
        print(f"❌ results differ from ground truth")
        return False

    print(f'Finished!')
    print('Layer 1 profiling:')
    print(net.layer1.get_profiling_stats())
    print('Layer 2 profiling:')
    print(net.layer2.get_profiling_stats())

    return True


def main():
    print("=" * 60)
    print("ANDNLayer FORWARD TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_andn_layer_forward(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
