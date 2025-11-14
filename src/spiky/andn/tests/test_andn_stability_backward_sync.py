import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import sys

from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from spiky.andn.ANDNLayer import Conv2DANDNLayer, Grid2DInhibitionLayer, SynapseMeta
from spiky.util.test_utils import lex_idx
from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper, InhibitionGrid2DHelper
from spiky.util.visual_helpers import grayscale_to_red_and_blue


def test_andn_stability_backward_sync(
    device, summation_dtype, seed=1
):
    if summation_dtype == torch.float32:
        return True
    if device == torch.device('cpu'):
        return True
    success = _test_andn_stability_backward_sync(0.4, 0.0, device, seed)
    return success


def _test_andn_stability_backward_sync(
    anti_hebb_coeff, initial_noise_level, device, seed=456678
):
    cpu_device = torch.device('cpu')
    summation_dtype = torch.int32
    torch.manual_seed(seed)
    input_shape = (28, 28)
    receptive_field_shape = (28, 28)
    receptive_field_stride_shape = (28, 28)
    output_kernel_shape = (4, 4)
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
            self._last_l1_output = None

        def forward(self, x):
            self._last_l1_output = self.layer1(x)
            return self.layer3(
                self.layer2(
                    self._last_l1_output
                ).reshape(
                    x.shape[0], output_kernel_shape[0] * output_kernel_shape[1]
                )
            )

    print(f'Creating TestNet, input_shape {input_shape}...')
    test_net_1 = TestNet(cpu_device)

    initial_weights = test_net_1.layer1._weights.data.clone().detach()
    test_net_2 = TestNet(device)
    test_net_2.layer1._weights.data = initial_weights.clone().to(device=device)

    mnist_dataset_dir = 'mnist'
    batch_size = 1
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

    optimizer_1 = optim.SGD(test_net_1.parameters(), 0.04)
    optimizer_2 = optim.SGD(test_net_2.parameters(), 0.04)
    loss_func = nn.CrossEntropyLoss()
    test_net_1.train()
    test_net_2.train()
    for batch_index, (data, target,) in tqdm(enumerate(test_batches)):
        weights_diff = (test_net_1.layer1._weights.data.detach() - test_net_2.layer1._weights.data.cpu()).abs().max().item()
        if weights_diff > 0.000001:
            print(f"❌ weights on CPU differs from weights on GPU, diff is {weights_diff}, batch_index is {batch_index}")
            return False

        optimizer_1.zero_grad()
        output_1 = test_net_1(data)
        loss = loss_func(output_1, target)
        loss.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        output_2 = test_net_2(data.to(device=device))
        loss = loss_func(output_2, target.to(device=device))
        loss.backward()
        optimizer_2.step()

        l1_output_diff = (test_net_1._last_l1_output - test_net_2._last_l1_output.cpu()).abs().max().item()
        if l1_output_diff > 0.000001:
            print(f"❌ l1_output_diff on CPU differs from l1_output_diff on GPU, diff is {l1_output_diff}, batch_index is {batch_index}")
            mask = (test_net_1._last_l1_output - test_net_2._last_l1_output.cpu()).abs() > 0.000000001
            print((test_net_1._last_l1_output - test_net_2._last_l1_output.cpu()).abs()[mask])
            return False

        last_w_grad_diff = (test_net_1.layer1._last_w_grad - test_net_2.layer1._last_w_grad.cpu()).abs().max().item()
        if last_w_grad_diff > 0.000001:
            print(f"❌ last_w_grad on CPU differs from last_w_grad on GPU, diff is {last_w_grad_diff}, batch_index is {batch_index}")
            return False

    return True


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if seed is None:
        seed = 1

    print("=" * 60)
    print("ANDNLayer STABILITY BACKWARD SYNC TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print('Skipping, this test is relevant only with cuda')
        return 0

    device = 'cuda'
    summation_dtype = torch.int32
    print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
    success = test_andn_stability_backward_sync(device, summation_dtype, seed=seed)

    if success:
        print(f"\n<{device}, {summation_dtype}> test completed successfully!")
    else:
        print(f"\n<{device}, {summation_dtype}> test failed!")
        return -1

    return 0


if __name__ == "__main__":
    exit(main())
