"""exp_c11 — torch reference for the two extensions (#75). SPIKY venv.

Emits, in fp32 with autocast off:
  * a HyperplaneMultiHeadLUT in hybrid_smooth mode (random hyperplanes), and
  * a FastMultiHeadLut (anchor pairs) in hard mode, together with the anchor indices,
    so the JAX side can build the equivalent frozen hyperplane w = e_a - e_b and check
    it reproduces the anchor-pair forward exactly.
"""
import argparse, os

import numpy as np
import torch

from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=32)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--input-dim", type=int, default=17)
    ap.add_argument("--outputs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2048)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    kw = dict(input_dim=a.input_dim, n_heads=a.heads, n_outputs=a.outputs,
              n_anchor_pairs=a.nap, tables_per_head=a.tph,
              weight_dtype=torch.float32, use_bf16=False,
              initial_weights_noise=0.05, learnable_temps=True, random_seed=0,
              device=dev)

    hp = HyperplaneMultiHeadLUT(forward_mode="hybrid_smooth",
                                hyperplane_init="anchor_pairs",
                                hyperplane_dtype=torch.float32, **kw).to(dev)
    with torch.no_grad():
        hp.hyperplane_weight.normal_(0.0, 0.5)
        hp.hyperplane_bias.normal_(0.0, 0.1)
    x = torch.randn(a.batch, a.input_dim, device=dev)
    with torch.no_grad():
        y_smooth = hp(x)

    fa = FastMultiHeadLut(forward_mode="hard", **kw).to(dev)
    with torch.no_grad():
        y_anchor = fa(x)

    np.savez(os.path.join(HERE, "torch_ext.npz"),
             w=hp.hyperplane_weight.detach().cpu().numpy(),
             b=hp.hyperplane_bias.detach().cpu().numpy(),
             weights=hp.weights.detach().cpu().numpy(),
             log_T_soft=hp.log_soft_score_temp.detach().cpu().numpy(),
             log_T_sel=hp.log_select_temp.detach().cpu().numpy(),
             x=x.cpu().numpy(), y_smooth=y_smooth.detach().cpu().numpy(),
             fa_weights=fa.weights.detach().cpu().numpy(),
             anchor_a=fa.soft_anchor_a_long.detach().cpu().numpy(),
             anchor_b=fa.soft_anchor_b_long.detach().cpu().numpy(),
             y_anchor=y_anchor.detach().cpu().numpy(),
             n_heads=np.int32(a.heads), tph=np.int32(a.tph))
    print(f"nap={a.nap} tph={a.tph} outputs={a.outputs} batch={a.batch} -> torch_ext.npz")


if __name__ == "__main__":
    main()
