"""exp_c04 — torch side of the JAX-port verification (#75). Run in the SPIKY venv.

Builds a HyperplaneMultiHeadLUT in **fp32, hard forward, autocast OFF** (the module's
own docstring notes parity is bit-exact only in that setting — under bf16 a
pre-activation near zero can flip a sign bit and select a different, equally correct
row, which no tolerance can close), runs it on a fixed random batch, and writes both
the weights and the reference output to an .npz for the JAX side to check against.

Usage:  python emit_torch_reference.py [--nap 8 --tph 64 --heads 1 --batch 4096]
"""
import argparse, os

import numpy as np
import torch

from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nap", type=int, default=8)
    ap.add_argument("--tph", type=int, default=64)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--input-dim", type=int, default=17)
    ap.add_argument("--outputs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "torch_reference.npz"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda"
    lut = HyperplaneMultiHeadLUT(
        input_dim=a.input_dim, n_heads=a.heads, n_outputs=a.outputs,
        n_anchor_pairs=a.nap, tables_per_head=a.tph,
        forward_mode="hard",                    # the deployable path
        hyperplane_init="anchor_pairs",         # == FastMultiHeadLut bit-for-bit
        weight_dtype=torch.float32, hyperplane_dtype=torch.float32,
        use_bf16=False,                         # REQUIRED for bit-exact parity
        initial_weights_noise=0.05, learnable_temps=True,
        random_seed=a.seed, device=dev).to(dev)

    # Randomise the hyperplanes away from the anchor-pair init so the test exercises
    # general affine addressing, not just the sparse +1/-1 special case.
    with torch.no_grad():
        lut.hyperplane_weight.normal_(0.0, 0.5)
        lut.hyperplane_bias.normal_(0.0, 0.1)

    x = torch.randn(a.batch, a.input_dim, device=dev, dtype=torch.float32)
    with torch.no_grad():
        y = lut(x)                              # [B, heads, outputs]

    np.savez(a.out,
             w=lut.hyperplane_weight.detach().float().cpu().numpy(),
             b=lut.hyperplane_bias.detach().float().cpu().numpy(),
             weights=lut.weights.detach().float().cpu().numpy(),
             n_heads=np.int32(a.heads), tph=np.int32(a.tph),
             obs_mean=np.zeros(a.input_dim, np.float32),
             obs_std=np.ones(a.input_dim, np.float32),
             x=x.cpu().numpy(), y=y.detach().float().cpu().numpy())
    print(f"nap={a.nap} tph={a.tph} heads={a.heads} batch={a.batch} "
          f"| w{tuple(lut.hyperplane_weight.shape)} "
          f"weights{tuple(lut.weights.shape)} y{tuple(y.shape)}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
