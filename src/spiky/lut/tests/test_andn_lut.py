import torch
import torch.nn as nn

from spiky.lut.ANDNLUTLayer import ANDNLUTLayerEx
from spiky.lut.LUTLayer import GradientPolicy, GradientType, LUTSharedContext
from spiky.andn.ANDNLayer import SynapseMeta


def main():
    device = 'cpu'
    summation_dtype = torch.float32
    random_seed = 1234
    input_shape = (28, 28)
    final_output_shape = (1, 10)
    synapse_meta_1 = SynapseMeta(
        min_weight=0.0,
        max_weight=10.0,
        initial_weight=10.0,
        initial_noise_level=-10.0
    )

    shared_lut_ctx = LUTSharedContext()
    shared_lut_ctx.to_device(device)
    g_policy = GradientPolicy(GradientType.Dense, normalized=False)

    class LUTANDNNet(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.layer1 = ANDNLUTLayerEx(
                input_shape=input_shape,
                output_shape=(input_shape[0] // 2, input_shape[1] // 2),
                n_anchors_per_detector=3,
                n_detector_groups=256,
                n_detectors_in_group=4,
                receptive_shape=(5, 5),
                projection_shape=(5, 5),
                n_projections_per_detector=32,
                inhibition_window_shape=(3, 3),
                n_inhibitors=256,
                n_neurons_per_inhibitor=4,
                synapse_meta=synapse_meta_1,
                backprop_hebb_ratio_on_torch_backward=0.0,
                anti_hebb_coeff=0.0,
                relu_before_inhibition=True,
                residual=False,
                weights_gradient_policy=g_policy,
                shared_context=shared_lut_ctx,
                summation_dtype=summation_dtype,
                random_seed=random_seed,
                device=device
            )

            osh = self.layer1._andn_layer.output_shape()
            self.final_layer = nn.Linear(osh[0] * osh[1], final_output_shape[1], device=device)

        def forward(self, x):
            return self.final_layer(
                self.layer1(
                    x.unsqueeze(1)
                ).view(x.shape[0], -1)
            )

    layered_andn_net = LUTANDNNet(device)
    # Count parameters
    lut_total_params = sum(p.numel() for p in layered_andn_net.parameters())
    lut_trainable_params = sum(p.numel() for p in layered_andn_net.parameters() if p.requires_grad)

    print(f"LUTNet Model: {layered_andn_net}")
    print(f"Total parameters: {lut_total_params:,}")
    print(f"Trainable parameters: {lut_trainable_params:,}")

    x = torch.rand([2, 28, 28])
    r = layered_andn_net(x)
    nn.CrossEntropyLoss()(r, torch.randint(10, [2])).backward()

    return 0


if __name__ == "__main__":
    exit(main())
