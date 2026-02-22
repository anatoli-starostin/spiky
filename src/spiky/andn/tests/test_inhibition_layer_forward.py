import torch
import torch.nn as nn

from spiky.andn.ANDNLayer import Grid2DInhibitionLayer
from spiky.util.synapse_growth import InhibitionGrid2DHelper


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


def test_inhibition_layer_forward(
    device, summation_dtype=torch.float32, seed=None
):
    if summation_dtype != torch.float32:
        return True
    success = _test_inhibition_layer_forward(
        input_shape=(4, 4),
        inhibition_grid_shape=(2, 2),
        device=device,
        seed=seed
    )
    success = success and _test_inhibition_layer_forward(
        input_shape=(16, 16),
        inhibition_grid_shape=(4, 4),
        device=device,
        seed=seed
    )
    return success


def _test_inhibition_layer_forward(
    input_shape,
    inhibition_grid_shape,
    device,
    batch_size=16,
    seed=None
):
    if seed is not None:
        torch.manual_seed(seed)

    class TestNet(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.layer1 = Grid2DInhibitionLayer(
                input_shape=input_shape,
                inhibition_grid_shape=inhibition_grid_shape,
                device=device
            )

        def forward(self, x):
            return self.layer1(x)

    print(f'Creating TestNet, input_shape {input_shape}...')
    net = TestNet(device)
    assert net.layer1.output_shape() == input_shape

    print(f'Creating GtNet, input_shape {input_shape}...')

    class GtNet(nn.Module):
        def __init__(
            self,
            input_shape,
            inhibition_grid_shape,
            device
        ):
            super().__init__()
            self.detectors = gt_detectors(input_shape, inhibition_grid_shape, device)

        def forward(self, x):
            y = torch.zeros_like(x)
            buckets = x[self.detectors]
            winners = buckets.argmax(dim=-1)
            y[self.detectors[torch.arange(buckets.size(0), device=x.device), winners]] = 1.0
            return y.reshape(input_shape)

    gt_net = GtNet(
        input_shape,
        inhibition_grid_shape,
        device
    )

    print(f'Calculating ground truth...')

    x = torch.rand([batch_size, input_shape[0] * input_shape[1]], device=device)
    gt_out = []
    for i in range(batch_size):
        gt_out.append(gt_net(x[i]))
    gt_out = torch.stack(gt_out)

    print(f'Calculating result...')

    y = net(x.reshape(batch_size, input_shape[0], input_shape[1]))

    if (gt_out - y).abs().max() > 0.001:
        print(f"❌ results differ from ground truth")
        return False

    print(f'Finished!')
    print('Layer 1 profiling:')
    print(net.layer1.get_profiling_stats())

    return True


def main():
    print("=" * 60)
    print("InhibitionLayer FORWARD TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        print(f"\nTesting on {device}...")
        success = test_inhibition_layer_forward(device)

        if success:
            print(f"\n<{device}> test completed successfully!")
        else:
            print(f"\n<{device}> test failed!")
            return -1
    return 0


if __name__ == "__main__":
    exit(main())
