import torch
import torch.nn.functional as nf
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from spiky.lut.LUTLayer import Conv2DLUTLayer, SynapseMeta, LUTLayer
from spiky.util.test_utils import lex_idx
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper


def test_lut_backward(
    device, summation_dtype, seed=123234
):
    success = _test_lut_backward(
        device=device,
        summation_dtype=summation_dtype,
        seed=seed
    )
    return success


def _test_lut_backward(
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
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                random_seed=seed,
                _int_rescaler=1.0,
                device=device
            )
            self.layer2 = nn.Linear(
                self.layer1.output_shape()[0] * self.layer1.output_shape()[1], 10, bias=False, device=device
            )

        def forward(self, x):
            x1 = self.layer1(x)
            return self.layer2(x1.reshape(x1.shape[0], x1.shape[-1] * x1.shape[-2]))

    print(f'Creating TestNet, input_shape {input_shape}...')
    test_net = TestNet(device)

    print(test_net)

    mnist_dataset_dir = 'mnist'
    batch_size = 64
    n_epochs = 4

    mnist_train_dataset = torchvision.datasets.MNIST(
        mnist_dataset_dir, train=True, download=True
    )
    mnist_train_data = mnist_train_dataset.data.to(device=torch.device('cpu')).to(dtype=torch.float32) / 255
    mnist_train_targets = mnist_train_dataset.targets.to(device=device)
    mnist_train_dataset = TensorDataset(
        (mnist_train_data / (mnist_train_data.norm(dim=(-1, -2), keepdim=True) + 1e-16)).to(device=device),
        mnist_train_targets
    )

    optimizer = optim.Adam(test_net.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    test_net.train()

    train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    pbar = tqdm(total=n_epochs * len(train_loader))
    for epoch in range(n_epochs):
        torch.manual_seed(seed + epoch)
        train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
        pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = test_net(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        train_acc = 100. * correct / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}, train accuracy: {train_acc:.2f}')
    pbar.close()
    print(f'Finished!')
    print('Layer 1 profiling:')
    print(test_net.layer1.get_profiling_stats())

    return True


def main():
    print("=" * 60)
    print("LUTLayer BACKWARD TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_lut_backward(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
