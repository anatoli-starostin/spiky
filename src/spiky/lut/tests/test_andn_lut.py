import torch
import torch.nn as nn
import torch.optim as optim

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
                output_shape=(input_shape[0] * 2, input_shape[1] * 2),
                n_anchors_per_detector=3,
                n_detector_groups=256,
                n_detectors_in_group=1,
                receptive_shape=(5, 5),
                projection_shape=(10, 10),
                projection_prob=0.5,
                inhibition_window_shape=(3, 3),
                do_normalize_weights=True,
                n_inhibitors=2048,
                n_neurons_per_inhibitor=9,
                dropout=0.0,
                synapse_meta=synapse_meta_1,
                backprop_hebb_ratio_on_torch_backward=0.5,
                anti_hebb_coeff=0.0,
                relu_before_inhibition=True,
                residual=False,
                weights_gradient_policy=g_policy,
                shared_context=shared_lut_ctx,
                summation_dtype=summation_dtype,
                random_seed=random_seed,
                device=device
            )

            self.layer2 = ANDNLUTLayerEx(
                input_shape=(input_shape[0] * 2, input_shape[1] * 2),
                output_shape=(input_shape[0], input_shape[1]),
                n_anchors_per_detector=5,
                n_detector_groups=64,
                n_detectors_in_group=16,
                receptive_shape=(20, 20),
                projection_shape=(5, 5),
                projection_prob=1.0,
                inhibition_window_shape=(3, 3),
                n_inhibitors=1024,
                n_neurons_per_inhibitor=9,
                dropout=0.0,
                synapse_meta=synapse_meta_1,
                backprop_hebb_ratio_on_torch_backward=0.5,
                anti_hebb_coeff=0.0,
                relu_before_inhibition=True,
                residual=False,
                weights_gradient_policy=g_policy,
                shared_context=shared_lut_ctx,
                summation_dtype=summation_dtype,
                random_seed=random_seed + 1,
                device=device
            )

            #         self.layer2 = LUTLayer(
            #             n_inputs=input_shape[0] * 2 * input_shape[1] * 2,
            #             n_outputs=64,
            #             n_detectors=256,
            #             n_anchors_per_detector=8,
            #             sequence_length=1,
            #             weights_gradient_policy=g_policy,
            #             shared_context=shared_lut_ctx,
            #             summation_dtype=summation_dtype,
            #             device=device,
            #             random_seed=random_seed,
            #         )

            osh = self.layer2._andn_layer.output_shape()
            # osh = (8, 8)
            self.final_layer = nn.Linear(osh[0] * osh[1], final_output_shape[1], device=device)
            self._l1_res = None
            self._l2_res = None
            # self.final_layer = nn.Linear(28 * 28, final_output_shape[1], device=device)

        def forward(self, x):
            self._l1_res = self.layer1(
                x.unsqueeze(1)
            )

            # self._l2_res = self.layer2(self._l1_res.reshape(x.shape[0], -1).unsqueeze(1))
            self._l2_res = self.layer2(self._l1_res.unsqueeze(1))

            return self.final_layer(
                self._l2_res.view(x.shape[0], -1)
            )

    layered_andn_net = LUTANDNNet(device)

    # Count parameters
    lut_total_params = sum(p.numel() for p in layered_andn_net.parameters())
    lut_trainable_params = sum(p.numel() for p in layered_andn_net.parameters() if p.requires_grad)

    print(f"LUTNet Model: {layered_andn_net}")
    print(f"Total parameters: {lut_total_params:,}")
    print(f"Trainable parameters: {lut_trainable_params:,}")

    optimizer = optim.Adam(layered_andn_net.parameters(), lr=0.001)

    x = torch.rand([32, 28, 28])
    optimizer.zero_grad()
    r = layered_andn_net(x)
    nn.CrossEntropyLoss()(r, torch.randint(10, [32])).backward()
    optimizer.step()
    return 0


if __name__ == "__main__":
    exit(main())
