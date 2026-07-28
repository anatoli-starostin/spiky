"""exp_c11 — torch reference GRADIENTS for the hybrid_smooth forward (#75). SPIKY venv.

fp32, autocast off. Emits torch's gradients w.r.t. x, hyperplane w/b, the table, and
both log-temperatures under `forward_mode="hybrid_smooth"`, so the JAX smooth backward
can be checked param-group by param-group.
"""
import argparse, os

import numpy as np
import torch

from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=32)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--input-dim", type=int, default=17)
    ap.add_argument("--outputs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-soft", type=float, default=None)
    ap.add_argument("--t-sel", type=float, default=None)
    ap.add_argument("--out", default="torch_smooth_grads.npz")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda"
    lut = HyperplaneMultiHeadLUT(
        input_dim=a.input_dim, n_heads=a.heads, n_outputs=a.outputs,
        n_anchor_pairs=a.nap, tables_per_head=a.tph,
        forward_mode="hybrid_smooth", hyperplane_init="anchor_pairs",
        weight_dtype=torch.float32, hyperplane_dtype=torch.float32,
        use_bf16=False, initial_weights_noise=0.05, learnable_temps=True,
        random_seed=a.seed, device=dev).to(dev)
    with torch.no_grad():
        lut.hyperplane_weight.normal_(0.0, 0.5)
        lut.hyperplane_bias.normal_(0.0, 0.1)
        if a.t_soft is not None:
            lut.log_soft_score_temp.fill_(np.log(a.t_soft))
        if a.t_sel is not None:
            lut.log_select_temp.fill_(np.log(a.t_sel))

    x = torch.randn(a.batch, a.input_dim, device=dev, requires_grad=True)
    g = torch.randn(a.batch, a.heads, a.outputs, device=dev)
    y = lut(x)
    y.backward(g)

    np.savez(os.path.join(HERE, a.out),
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
             grad_log_T_sel=lut.log_select_temp.grad.detach().cpu().numpy())
    print(f"nap={a.nap} tph={a.tph} batch={a.batch} T_soft={a.t_soft} "
          f"T_sel={a.t_sel} -> {a.out}")


if __name__ == "__main__":
    main()
