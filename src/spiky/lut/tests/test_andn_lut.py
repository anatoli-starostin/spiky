import torch
import torch.nn as nn

from spiky.lut.LUTLayer import Conv2DLUTLayer, SynapseMeta, LUTLayerBasic
from spiky.util.test_utils import lex_idx
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper


def create_connections(
    input_shape,
    receptive_field_shape,
    receptive_field_stride_shape,
    output_kernel_shape,
    device,
    n_input_channels=None
):
    c_helper = Conv2DSynapseGrowthHelper(
        input_shape[0], input_shape[1],
        receptive_field_shape[0], receptive_field_shape[1],
        receptive_field_stride_shape[0], receptive_field_stride_shape[1],
        output_kernel_shape[0], output_kernel_shape[1],
        n_input_channels=n_input_channels
    )

    if n_input_channels is None:
        n_input_channels = 1

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
                            for c in range(n_input_channels):
                                target_id = c_helper.h * c_helper.w * n_input_channels + (oy + by) * c_helper.out_w + ox + bx
                                ground_truth_pairs.add((source_id * n_input_channels + c, target_id,))

    source_ids = torch.zeros([len(ground_truth_pairs)], dtype=torch.int32, device=torch.device('cpu'))
    targets_ids = torch.zeros([len(ground_truth_pairs)], dtype=torch.int32, device=torch.device('cpu'))

    for i, (source_id, target_id) in enumerate(sorted(ground_truth_pairs)):
        source_ids[i] = source_id
        targets_ids[i] = target_id

    return source_ids.to(device), targets_ids.to(device)


def connections_to_matrix(n_inputs, n_outputs, source_ids, targets_ids, device):
    res = torch.zeros(n_inputs, n_outputs, dtype=torch.float32, device=device)
    res[source_ids, targets_ids - n_inputs] = 1.0
    return res


def test_lut_forward_simple(
    device, summation_dtype, seed=None
):
    success = _test_lut_forward_simple(
        input_shape=(4, 4),
        receptive_field_shape=(2, 2),
        receptive_field_stride_shape=(1, 1),
        lut_receptive_field_shape=(4, 4),
        lut_receptive_field_stride_shape=(2, 2),
        detectors_shape=(2, 2),
        output_kernel_shape=(2, 2),
        n_anchors_per_detector=2,
        synapse_group_size=3,
        device=device,
        summation_dtype=summation_dtype,
        seed=seed
    )
    success = success and _test_lut_forward_simple(
        input_shape=(28, 28),
        receptive_field_shape=(5, 5),
        receptive_field_stride_shape=(1, 1),
        detectors_shape=(2, 2),
        output_kernel_shape=(8, 8),
        n_anchors_per_detector=4,
        device=device,
        summation_dtype=summation_dtype,
        seed=seed
    )
    return success


def _test_lut_forward_simple(
    input_shape,
    receptive_field_shape,
    receptive_field_stride_shape,
    detectors_shape,
    output_kernel_shape,
    n_anchors_per_detector,
    device, summation_dtype,
    lut_receptive_field_shape=None,
    lut_receptive_field_stride_shape=None,
    synapse_group_size=64,
    batch_size=4, seed=None
):
    if seed is not None:
        torch.manual_seed(seed)
    synapse_meta = SynapseMeta(
        initial_weight=0.0,
        initial_noise_level=1.0
    )

    class TestNet(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.layer1 = Conv2DLUTLayer(
                input_shape=input_shape,
                n_anchors_per_detector=n_anchors_per_detector,
                detectors_shape=detectors_shape,
                output_kernel_shape=output_kernel_shape,
                sequence_length=1,
                receptive_field_shape=receptive_field_shape,
                receptive_field_stride_shape=receptive_field_stride_shape,
                lut_receptive_field_shape=lut_receptive_field_shape,
                lut_receptive_field_stride_shape=lut_receptive_field_stride_shape,
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                _int_rescaler=1.0,
                _forward_group_size=synapse_group_size,
                random_seed=seed,
                device=device
            )

        def forward(self, x):
            return self.layer1(x)

    print(f'Creating TestNet, input_shape {input_shape}...')
    net = TestNet(device)
    lut_shape = net.layer1.lut_shape()

    assert lut_shape == (
        (((input_shape[0] - receptive_field_shape[0]) // receptive_field_stride_shape[0]) + 1) * detectors_shape[0],
        (((input_shape[1] - receptive_field_shape[1]) // receptive_field_stride_shape[1]) + 1) * detectors_shape[1],
        LUTLayerBasic.n_lut_channels(n_anchors_per_detector, 1)
    )

    if lut_receptive_field_shape is None:
        lut_receptive_field_shape = lut_shape
        lut_receptive_field_stride_shape = lut_shape

    anchors = net.layer1._export_anchors()

    potential_connections = connections_to_matrix(
        input_shape[0] * input_shape[1],
        lut_shape[0] * lut_shape[1],
        *create_connections(
            input_shape,
            receptive_field_shape,
            receptive_field_stride_shape,
            detectors_shape,
            device
        ), device
    )

    assert torch.all(potential_connections.sum(dim=0) == receptive_field_shape[0] * receptive_field_shape[1])

    for detector_id in range(anchors.shape[0]):
        for i in range(anchors.shape[1]):
            anchor_id1 = anchors[detector_id, i, 0]
            anchor_id2 = anchors[detector_id, i, 1]
            if potential_connections[anchor_id1, detector_id] != 1.0:
                print(f"❌ wrong anchor_id1 {anchor_id1} found for detector {detector_id}")
                return False

            if potential_connections[anchor_id2, detector_id] != 1.0:
                print(f"❌ wrong anchor_id2 {anchor_id2} found for detector {detector_id}")
                return False

    layer_1_output_shape = (
        (((lut_shape[0] - lut_receptive_field_shape[0]) // lut_receptive_field_stride_shape[0]) + 1) * output_kernel_shape[0],
        (((lut_shape[1] - lut_receptive_field_shape[1]) // lut_receptive_field_stride_shape[1]) + 1) * output_kernel_shape[1]
    )

    assert net.layer1.output_shape() == layer_1_output_shape

    print(f'Creating gt_connections, layer 1...')

    layer_1_gt_source_ids, layer_1_gt_target_ids = create_connections(
        input_shape=lut_shape[:2],
        receptive_field_shape=lut_receptive_field_shape,
        receptive_field_stride_shape=lut_receptive_field_stride_shape,
        output_kernel_shape=output_kernel_shape,
        n_input_channels=LUTLayerBasic.n_lut_channels(n_anchors_per_detector, 1),
        device=device
    )

    print(net.layer1)

    print(f'Exporting synapses, layer 1...')

    n_synapses = net.layer1._count_synapses(net.layer1.get_lookup_neuron_ids())
    synapses_export = {
        'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
        'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
    }

    net.layer1._export_synapses(
        net.layer1.get_lookup_neuron_ids(),
        synapses_export['source_ids'],
        synapses_export['weights'],
        synapses_export['target_ids'],
    )

    order1 = lex_idx(synapses_export['source_ids'], synapses_export['target_ids'])

    if torch.any(synapses_export['source_ids'][order1] != layer_1_gt_source_ids):
        print(f"❌ wrong source connections at layer 1")
        return False

    if torch.any(synapses_export['target_ids'][order1] != layer_1_gt_target_ids):
        print(f"❌ wrong target connections at layer 1")
        return False

    x = torch.rand([batch_size, 1, input_shape[0], input_shape[1]], device=device)
    y = net(x)

    print(y)
    print(f'Finished!')
    print('Layer 1 profiling:')
    print(net.layer1.get_profiling_stats())

    return True


def main():
    print("=" * 60)
    print("LUTLayer FORWARD SIMPLE TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_lut_forward_simple(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
