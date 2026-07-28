"""exp_c06 — torch reference GRADIENTS for the backward-port check (#75). SPIKY venv.

fp32, hard forward, autocast off (the only regime where parity is bit-exact — see
the module docstring on bf16 sign flips). Emits weights, input, upstream grad, and
torch's gradients w.r.t. x, hyperplane w/b, the table, and both log-temperatures.
"""
import argparse, os

import numpy as np
import torch

from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=16)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--input-dim", type=int, default=17)
    ap.add_argument("--outputs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "torch_grads.npz"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda"
    lut = HyperplaneMultiHeadLUT(
        input_dim=a.input_dim, n_heads=a.heads, n_outputs=a.outputs,
        n_anchor_pairs=a.nap, tables_per_head=a.tph,
        forward_mode="hard", hyperplane_init="anchor_pairs",
        weight_dtype=torch.float32, hyperplane_dtype=torch.float32,
        use_bf16=False, initial_weights_noise=0.05, learnable_temps=True,
        random_seed=a.seed, device=dev).to(dev)
    with torch.no_grad():
        lut.hyperplane_weight.normal_(0.0, 0.5)
        lut.hyperplane_bias.normal_(0.0, 0.1)

    x = torch.randn(a.batch, a.input_dim, device=dev, requires_grad=True)
    g = torch.randn(a.batch, a.heads, a.outputs, device=dev)

    y = lut(x)
    y.backward(g)

    np.savez(
        a.out,
        w=lut.hyperplane_weight.detach().cpu().numpy(),
        b=lut.hyperplane_bias.detach().cpu().numpy(),
        weights=lut.weights.detach().cpu().numpy(),
        log_T_soft=lut.log_soft_score_temp.detach().cpu().numpy(),
        log_T_sel=lut.log_select_temp.detach().cpu().numpy(),
        n_heads=np.int32(a.heads), tph=np.int32(a.tph),
        x=x.detach().cpu().numpy(), g=g.cpu().numpy(),
        y=y.detach().cpu().numpy(),
        grad_x=x.grad.detach().cpu().numpy(),
        grad_w=lut.hyperplane_weight.grad.detach().cpu().numpy(),
        grad_b=lut.hyperplane_bias.grad.detach().cpu().numpy(),
        grad_weights=lut.weights.grad.detach().cpu().numpy(),
        grad_log_T_soft=lut.log_soft_score_temp.grad.detach().cpu().numpy(),
        grad_log_T_sel=lut.log_select_temp.grad.detach().cpu().numpy(),
    )
    print(f"nap={a.nap} tph={a.tph} heads={a.heads} batch={a.batch} -> {a.out}")
    print(f"  |grad_x| {x.grad.abs().max():.5f}  |grad_w| "
          f"{lut.hyperplane_weight.grad.abs().max():.5f}  |grad_weights| "
          f"{lut.weights.grad.abs().max():.5f}")


if __name__ == "__main__":
    main()
