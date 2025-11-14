import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from spiky.andn.ANDNLayer import Conv2DANDNLayer, Grid2DInhibitionLayer, SynapseMeta
from spiky.util.test_utils import lex_idx
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper, InhibitionGrid2DHelper
from spiky.util.visual_helpers import grayscale_to_red_and_blue


def test_andn_stability_backward(
    device, summation_dtype, seed=1
):
    if summation_dtype == torch.float32:
        return True
    if device == torch.device('cpu'):
        return True
    success = _test_andn_stability_backward(0.0, 0.0, device, seed)
    success = success and _test_andn_stability_backward(0.4, 0.0, device, seed)
    success = success and _test_andn_stability_backward(0.0, -1.0, device, seed)
    success = success and _test_andn_stability_backward(0.4, -1.0, device, seed)
    return success


def _test_andn_stability_backward(
    anti_hebb_coeff, initial_noise_level, device, seed=456678
):
    cpu_device = torch.device('cpu')
    summation_dtype = torch.int32
    torch.manual_seed(seed)
    input_shape = (28, 28)
    receptive_field_shape = (28, 28)
    receptive_field_stride_shape = (28, 28)
    output_kernel_shape = (8, 8)
    torch.manual_seed(seed)
    synapse_meta = SynapseMeta(
        min_weight=0.0,
        max_weight=1.0,
        initial_weight=1.0,
        initial_noise_level=initial_noise_level
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
                anti_hebb_coeff=anti_hebb_coeff,
                summation_dtype=summation_dtype,
                random_seed=seed,
                device=device
            )
            self.layer2 = Grid2DInhibitionLayer(
                input_shape=output_kernel_shape,
                inhibition_grid_shape=output_kernel_shape,
                device=device
            )
            self.layer1.set_descendant_andn_layer(self.layer2)
            self.layer3 = nn.Linear(output_kernel_shape[0] * output_kernel_shape[1], 10, device=device)

        def forward(self, x):
            return self.layer3(
                self.layer2(
                    self.layer1(x)
                ).reshape(
                    x.shape[0], output_kernel_shape[0] * output_kernel_shape[1]
                )
            )

    print(f'Creating TestNet, input_shape {input_shape}...')
    test_net = TestNet(cpu_device)

    assert test_net.layer1.output_shape() == output_kernel_shape
    gt_initial_weights = test_net.layer1._weights.data.clone().detach()

    mnist_dataset_dir = 'mnist'
    batch_size = 32
    n_batches = 1000

    mnist_train_dataset = torchvision.datasets.MNIST(
        mnist_dataset_dir, train=True, download=True
    )
    mnist_train_data = mnist_train_dataset.data.to(device=cpu_device).to(dtype=torch.float32) / 255
    mnist_train_targets = mnist_train_dataset.targets.to(device=cpu_device)
    mnist_train_dataset = TensorDataset(
        mnist_train_data, mnist_train_targets
    )
    test_batches = []
    train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    for data, target in train_loader:
        test_batches.append((data / (data.norm(dim=(-1, -2), keepdim=True) + 1e-16), target,))
        if len(test_batches) == n_batches:
            break

    optimizer = optim.SGD(test_net.parameters(), 0.04)
    loss_func = nn.CrossEntropyLoss()
    test_net.train()
    pbar = tqdm(total=n_batches)
    pbar.set_description(f'Preparing "ground truth" on CPU')
    for data, target in test_batches:
        optimizer.zero_grad()
        output = test_net(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        pbar.update(1)
    pbar.close()

    gt_weights = test_net.layer1.export_weights()

    for i, (d, t) in enumerate(test_batches):
        test_batches[i] = d.to(device=device), t.to(device=device)

    test_net = TestNet(device)
    test_net.layer1._weights.data = gt_initial_weights.clone().to(device=device)

    optimizer = optim.SGD(test_net.parameters(), 0.04)
    loss_func = nn.CrossEntropyLoss()
    test_net.train()
    pbar = tqdm(total=n_batches)
    pbar.set_description(f"Processing same data on GPU")
    for data, target in test_batches:
        optimizer.zero_grad()
        output = test_net(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        pbar.update(1)
    pbar.close()

    diff = (gt_weights - test_net.layer1.export_weights().cpu()).abs().max().item()
    if diff > 0.000001:
        print(f"❌ weights on CPU differs from weights on GPU, diff is {diff}")
        return False

    return True


def main():
    print("=" * 60)
    print("ANDNLayer STABILITY BACKWARD TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print('Skipping, this test is relevant only with cuda')
        return 0

    device = 'cuda'
    summation_dtype = torch.int32
    print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
    success = test_andn_stability_backward(device, summation_dtype)

    if success:
        print(f"\n<{device}, {summation_dtype}> test completed successfully!")
    else:
        print(f"\n<{device}, {summation_dtype}> test failed!")
        return -1

    return 0


if __name__ == "__main__":
    exit(main())
