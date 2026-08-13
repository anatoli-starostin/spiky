"""exp_c11 — torch fixture for the hyperplane anchor_pairs init check (#75). SPIKY venv.

Builds a HyperplaneMultiHeadLUT with its DEFAULT init (hyperplane_init="anchor_pairs",
policy CANONICAL_FULL_COVERAGE) and a FastMultiHeadLut forced onto the SAME anchor pairs,
then dumps both forwards plus the raw hyperplane parameters. The JAX side must reproduce
all of it.

Both modules are built on CPU: the anchor draw is device-dependent (lutorch seeds a
torch.Generator(device=...)), and cpu is what the trainer's cache uses by default.
"""
import argparse, os

import numpy as np
import torch

from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=32)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--input-dim", type=int, default=17)
    ap.add_argument("--outputs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device(a.device)
    policy = AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE
    kw = dict(input_dim=a.input_dim, n_heads=a.heads, n_outputs=a.outputs,
              n_anchor_pairs=a.nap, tables_per_head=a.tph,
              weight_dtype=torch.float32, use_bf16=False,
              initial_weights_noise=0.05, learnable_temps=True, random_seed=0,
              anchor_sampling_policy=policy, device=dev)

    # DEFAULT init -- hyperplane_init is not passed, so this is torch's own choice.
    hp = HyperplaneMultiHeadLUT(forward_mode="hard",
                                hyperplane_dtype=torch.float32, **kw).to(dev)
    assert hp.hyperplane_init == "anchor_pairs", hp.hyperplane_init

    fa = FastMultiHeadLut(forward_mode="hard", **kw).to(dev)
    # Force the two modules onto the SAME table weights so the only thing under test is
    # the addressing. Their anchor buffers already agree (same policy, seed, device).
    with torch.no_grad():
        fa.weights.copy_(hp.weights)
    assert torch.equal(fa.soft_anchor_a_long, hp.soft_anchor_a_long)
    assert torch.equal(fa.soft_anchor_b_long, hp.soft_anchor_b_long)

    x = torch.randn(a.batch, a.input_dim, device=dev)
    with torch.no_grad():
        y_hp, y_fa = hp(x), fa(x)

    np.savez(os.path.join(HERE, "torch_hp_init.npz"),
             hp_w=hp.hyperplane_weight.detach().cpu().numpy(),
             hp_b=hp.hyperplane_bias.detach().cpu().numpy(),
             weights=hp.weights.detach().cpu().numpy(),
             log_T_soft=hp.log_soft_score_temp.detach().cpu().numpy(),
             log_T_sel=hp.log_select_temp.detach().cpu().numpy(),
             anchor_a=hp.soft_anchor_a_long.cpu().numpy(),
             anchor_b=hp.soft_anchor_b_long.cpu().numpy(),
             x=x.cpu().numpy(), y_hp=y_hp.cpu().numpy(), y_fa=y_fa.cpu().numpy(),
             policy=policy.value, device=a.device,
             nap=np.int32(a.nap), input_dim=np.int32(a.input_dim),
             n_heads=np.int32(a.heads), tph=np.int32(a.tph))
    same = torch.equal(y_hp, y_fa)
    print(f"nap={a.nap} tph={a.tph} policy={policy.value} device={a.device} | "
          f"torch hp == torch fa at init: {same} -> torch_hp_init.npz")


if __name__ == "__main__":
    main()
