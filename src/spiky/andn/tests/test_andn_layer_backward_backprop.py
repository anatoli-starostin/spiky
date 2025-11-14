import torch
import torch.nn as nn
import torch.nn.functional as nf
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

cpu_gt_weights = None


def test_andn_layer_backward_backprop(
    device, summation_dtype, seed=5675
):
    success = _test_andn_layer_backward_backprop(device, summation_dtype, seed)
    return success


def _test_andn_layer_backward_backprop(
    device, summation_dtype, seed=4578
):
    torch.manual_seed(seed)
    input_shape = (28, 28)
    final_output_shape = (1, 10)
    receptive_field_shape = (28, 28)
    receptive_field_stride_shape = (28, 28)
    output_kernel_shape = (8, 8)
    torch.manual_seed(seed)
    synapse_meta = SynapseMeta(
        min_weight=0.0,
        max_weight=1.0,
        initial_weight=1.0,
        initial_noise_level=-1.0
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
                backprop_hebb_ratio_on_torch_backward=0.5,
                anti_hebb_coeff=0.4,
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                random_seed=seed,
                device=device
            )
            self.layer2 = Conv2DANDNLayer(
                input_shape=self.layer1.output_shape(),
                inhibition_grid_shape=output_kernel_shape,
                receptive_field_shape=self.layer1.output_shape(),
                receptive_field_stride_shape=self.layer1.output_shape(),
                output_kernel_shape=final_output_shape,
                backprop_hebb_ratio_on_torch_backward=0.0,
                synapse_meta=synapse_meta,
                summation_dtype=summation_dtype,
                random_seed=seed,
                device=device
            )
            self.layer1.set_descendant_andn_layer(self.layer2)

        def forward(self, x):
            return self.layer2(self.layer1(x)).squeeze(1)

    print(f'Creating TestNet, input_shape {input_shape}...')
    test_net = TestNet(device)
    print(test_net.layer1)

    if summation_dtype == torch.int32:
        global cpu_gt_weights
        if device == torch.device('cpu'):
            if cpu_gt_weights is None:
                cpu_gt_weights = test_net.layer1._weights.data.clone().detach()
        elif cpu_gt_weights is not None:
            test_net.layer1._weights.data = cpu_gt_weights.clone().to(device=device)

    assert test_net.layer1.output_shape() == output_kernel_shape

    mnist_dataset_dir = 'mnist'
    batch_size = 256
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

    optimizer = optim.SGD(test_net.parameters(), 0.01)
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

    target_image = torch.zeros([output_kernel_shape[0] * (input_shape[0] + 2), output_kernel_shape[1] * (input_shape[1] + 2)])
    weights = test_net.layer1.export_weights()
    weights /= weights.abs().max()

    for i in range(output_kernel_shape[0]):
        for j in range(output_kernel_shape[1]):
            target_image[
                i * (input_shape[0] + 2) + 1:i * (input_shape[0] + 2) + input_shape[0] + 1,
                j * (input_shape[1] + 2) + 1:j * (input_shape[1] + 2) + input_shape[1] + 1
            ] = weights[i, j]

    img = grayscale_to_red_and_blue(target_image.unsqueeze(0))
    color_rescaler = 1.0 / (img.abs().max() + 0.0000000001)
    img = to_pil_image((img * color_rescaler).clip(0.0, 1.0))
    conf_name = f'device_{device}_sum_dt_{summation_dtype}'
    img.save(f'andn_layer_backward_backprop_test_{conf_name}_color_rescale_{color_rescaler:.2}.png')

    print('Layer 1 profiling:')
    print(test_net.layer1.get_profiling_stats())

    return True


def main():
    print("=" * 60)
    print("ANDNLayer BACKWARD BACKPROP TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_andn_layer_backward_backprop(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
