import torch
import torch.nn.functional as nf
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from spiky.lut.LUTLayer import Conv2DLUTLayer, SynapseMeta
from spiky.util.test_utils import lex_idx
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper


def test_lut_fully_connected(
    device, summation_dtype, seed=123234
):
    success = _test_lut_fully_connected(
        device=device,
        summation_dtype=summation_dtype,
        seed=seed
    )
    return success


def _test_lut_fully_connected(
    device, summation_dtype, seed=123243
):
    torch.manual_seed(seed)

    input_shape = (28, 28)
    receptive_field_shape = (28, 28)
    receptive_field_stride_shape = (1, 1)
    detectors_shape = (8, 8)
    output_kernel_shape = (16, 16)
    n_anchors_per_detector = 5

    synapse_meta = SynapseMeta(
        initial_weight=0.0,
        initial_noise_level=1.0
    )

    class TestNet(nn.Module):
        def __init__(self, device, lut_receptive_field_shape):
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
                lut_receptive_field_stride_shape=lut_receptive_field_shape,
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                random_seed=seed,
                _int_rescaler=1.0,
                device=device
            )
            self.layer1._set_lookup_inidices_callback(self.store_lookup_indices)
            self.layer2 = nn.Linear(
                self.layer1.output_shape()[0] * self.layer1.output_shape()[1], 10, bias=False, device=device
            )
            self._last_hidden_output = None
            self._last_lookup_indices = None

        def store_lookup_indices(self, lookup_inidices, _, __):
            self._last_lookup_indices = lookup_inidices.detach()

        def forward(self, x):
            x1 = self.layer1(x)
            self._last_hidden_output = x1
            return self.layer2(x1.reshape(x1.shape[0], x1.shape[-1] * x1.shape[-2]))

    print(f'Creating TestNet, input_shape {input_shape}...')
    test_net_standard = TestNet(device, detectors_shape)
    test_net_fully_connected = TestNet(device, None)

    weights_standard = test_net_standard.layer1.export_weights(inverse_order=False)
    with torch.no_grad():
        test_net_fully_connected.layer1._weights[:] = weights_standard.flatten()
        test_net_fully_connected.layer2.weight[:] = test_net_standard.layer2.weight.detach()

    weights_standard = test_net_standard.layer1.export_weights(inverse_order=False)
    weights = test_net_fully_connected.layer1.export_weights(inverse_order=False)

    if (weights_standard - weights).abs().max() > 0.00001:
        print(f"❌ weights difference detected before training has started")
        return False

    anchors_standard = test_net_standard.layer1._export_anchors()
    anchors = test_net_fully_connected.layer1._export_anchors()

    if (anchors_standard - anchors).sum() != 0:
        print(f"❌ anchors difference detected")
        return False

    print(test_net_standard)
    print(test_net_fully_connected)

    mnist_dataset_dir = 'mnist'
    batch_size = 128
    n_epochs = 1

    mnist_train_dataset = torchvision.datasets.MNIST(
        mnist_dataset_dir, train=True, download=True
    )
    mnist_train_data = mnist_train_dataset.data.to(device=torch.device('cpu')).to(dtype=torch.float32) / 255
    mnist_train_targets = mnist_train_dataset.targets.to(device=device)
    mnist_train_dataset = TensorDataset(
        (mnist_train_data / (mnist_train_data.norm(dim=(-1, -2), keepdim=True) + 1e-16)).to(device=device),
        mnist_train_targets
    )

    optimizer = optim.Adam(test_net_fully_connected.parameters(), lr=0.001)
    optimizer_standard = optim.Adam(test_net_standard.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    test_net_fully_connected.train()
    test_net_standard.train()

    train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    pbar = tqdm(total=n_epochs * len(train_loader))
    for epoch in range(n_epochs):
        torch.manual_seed(seed + epoch)
        train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
        pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
        correct = 0
        for data, target in train_loader:
            optimizer_standard.zero_grad()
            output = test_net_standard(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer_standard.step()

            optimizer.zero_grad()
            output = test_net_fully_connected(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            diff = (test_net_standard._last_lookup_indices - test_net_fully_connected._last_lookup_indices).sum()
            if diff != 0:
                print(f"❌ lookup_indices difference detected, diff {diff}")
                print('test_net_standard._last_lookup_indices:\n')
                print(test_net_standard._last_lookup_indices)
                print('test_net_fully_connected._last_lookup_indices:\n')
                print(test_net_fully_connected._last_lookup_indices)
                return False

            diff = (test_net_standard._last_hidden_output - test_net_fully_connected._last_hidden_output).abs().max()
            if diff > 0.001:
                print(f"❌ hidden output difference detected, diff {diff}")
                print('test_net_standard._last_hidden_output:\n')
                print(test_net_standard._last_hidden_output)
                print('test_net_fully_connected._last_hidden_output:\n')
                print(test_net_fully_connected._last_hidden_output)
                return False

            weights_standard = test_net_standard.layer1.export_weights(inverse_order=False)
            # weights = test_net_fully_connected.layer1.export_weights(inverse_order=False)
            #
            # if (weights_standard - weights).abs().max() > 0.001:
            #     print(f"❌ weights difference detected")
            #     return False

            with torch.no_grad():
                test_net_fully_connected.layer1._weights[:] = weights_standard.flatten()
                test_net_fully_connected.layer2.weight[:] = test_net_standard.layer2.weight.detach()

        train_acc = 100. * correct / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}, train accuracy: {train_acc:.2f}')
    pbar.close()
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

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_lut_fully_connected(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
