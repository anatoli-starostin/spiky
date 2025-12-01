import torch
import torch.nn.functional as nf
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from spiky.lut.LUTLayer import LUTLayerBasic, LUTLayer, Conv2DLUTLayer, SynapseMeta
from spiky.util.test_utils import lex_idx
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper


def test_lut_fully_connected_small(
    device, summation_dtype, seed=123234
):
    success = _test_lut_fully_connected_small(
        n_inputs=32,
        n_anchors_per_detector=3,
        n_detectors=4,
        n_outputs=16,
        batch_size=16,
        device=device,
        summation_dtype=summation_dtype,
        seed=seed
    )
    return success


def _test_lut_fully_connected_small(
    n_inputs,
    n_anchors_per_detector,
    n_detectors,
    n_outputs,
    batch_size,
    device, summation_dtype, seed=123243
):
    torch.manual_seed(seed)

    synapse_meta = SynapseMeta(
        initial_weight=0.0,
        initial_noise_level=1.0
    )

    class TestNet(nn.Module):
        def __init__(self, device, is_fully_connected):
            super().__init__()
            if is_fully_connected:
                self.layer1 = LUTLayer(
                    n_inputs=n_inputs,
                    n_anchors_per_detector=n_anchors_per_detector,
                    n_detectors=n_detectors,
                    n_outputs=n_outputs,
                    synapse_meta=synapse_meta,
                    summation_dtype=summation_dtype,
                    random_seed=seed,
                    _int_rescaler=1.0,
                    device=device
                )
            else:
                self.layer1 = Conv2DLUTLayer(
                    input_shape=(1, n_inputs),
                    n_anchors_per_detector=n_anchors_per_detector,
                    detectors_shape=(1, n_detectors),
                    output_kernel_shape=(1, n_outputs),
                    receptive_field_shape=(1, n_inputs),
                    receptive_field_stride_shape=(1, n_inputs),
                    lut_receptive_field_shape=(1, n_detectors),
                    lut_receptive_field_stride_shape=(1, n_detectors),
                    synapse_meta=synapse_meta,
                    summation_dtype=summation_dtype,
                    random_seed=seed,
                    _int_rescaler=1.0,
                    device=device
                )

        def forward(self, x):
            return self.layer1(x.unsqueeze(1)).squeeze(1)

    print(f'Creating TestNet...')
    test_net_standard = TestNet(device, False)
    test_net_fully_connected = TestNet(device, True)

    weights_standard = test_net_standard.layer1.export_weights(inverse_order=False)
    with torch.no_grad():
        test_net_fully_connected.layer1._weights[:] = weights_standard.flatten()

    weights_standard = test_net_standard.layer1.export_weights(inverse_order=False).reshape(n_detectors, LUTLayerBasic.n_lut_channels(n_anchors_per_detector, 1), n_outputs)
    weights = test_net_fully_connected.layer1.export_weights(inverse_order=False)

    if (weights_standard - weights).abs().max() > 0.00001:
        print(f"❌ weights difference detected before training had started")
        return False
    
    print(test_net_standard)
    print(test_net_fully_connected)

    optimizer = optim.Adam(test_net_fully_connected.parameters(), lr=0.001)
    optimizer_standard = optim.Adam(test_net_standard.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    test_net_fully_connected.train()
    test_net_standard.train()

    for _ in tqdm(range(1000)):
        data = torch.rand([batch_size, n_inputs], device=device)
        target = torch.rand([batch_size, n_outputs], device=device)
        optimizer_standard.zero_grad()
        output_standard = test_net_standard(data.reshape(batch_size, 1, n_inputs)).reshape(batch_size, n_outputs)
        loss = loss_func(output_standard, target)
        loss.backward()
        optimizer_standard.step()

        optimizer.zero_grad()
        output = test_net_fully_connected(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        if (output_standard - output).abs().max() > 0.00001:
            print(f"❌ output difference detected")
            return False

        weights_standard = test_net_standard.layer1.export_weights(inverse_order=False).reshape(n_detectors, LUTLayerBasic.n_lut_channels(n_anchors_per_detector, 1), n_outputs)
        weights = test_net_fully_connected.layer1.export_weights(inverse_order=False)

        if (weights_standard - weights).abs().max() > 0.00001:
            print(f"❌ weights difference detected")
            return False

        with torch.no_grad():
            test_net_fully_connected.layer1._weights[:] = weights_standard.flatten()

    print(f'Finished!')
    print('Layer 1 standard profiling:')
    print(test_net_standard.layer1.get_profiling_stats())
    print('Layer 1 fully connected profiling:')
    print(test_net_fully_connected.layer1.get_profiling_stats())

    return True


def main():
    print("=" * 60)
    print("LUTLayer FULLY CONNECTED TEST")
    print("=" * 60)

    devices = []
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_lut_fully_connected_small(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
