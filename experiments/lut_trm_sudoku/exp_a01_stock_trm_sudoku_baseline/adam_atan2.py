"""Pure-PyTorch drop-in replacement for the fused `adam-atan2` CUDA kernel.

WHY THIS EXISTS (exp_a01, Phase A of #73): the PyPI `adam-atan2` package ships a fused
CUDA kernel that (1) fails to compile against torch 2.9.1's headers under nvcc+gcc on this
box (a `typename`/dependent-scope error in ATen/core/List_inl.h, unrelated to the GPU arch),
and (2) hardcodes `code=sm_80/86/89/90` with no PTX, so it would not even run on the RTX 5090
(sm_120) if it built. This module reproduces the *exact same optimizer math* in pure PyTorch,
so `pretrain.py`'s `from adam_atan2 import AdamATan2` works unmodified.

The update is Adam with an epsilon-free, bounded step:

    m_t   = b1 * m_{t-1} + (1 - b1) * g
    v_t   = b2 * v_{t-1} + (1 - b2) * g^2
    m_hat = m_t / (1 - b1^t)
    v_hat = v_t / (1 - b2^t)
    p    <- p - lr * a * atan2(m_hat, b * sqrt(v_hat))     # replaces m_hat/(sqrt(v_hat)+eps)

with decoupled (AdamW-style) weight decay. Defaults a=1.0, b=1.0 match the fused kernel, so
this is mathematically identical (only unfused → marginally lower throughput; accuracy
unchanged). See the Adam-atan2 optimizer (epsilon-free Adam) from Everett et al.,
"Scaling Exponents Across Parameterizations and Optimizers" (2024).
"""
import torch
from torch.optim.optimizer import Optimizer


class AdamATan2(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, a=1.0, b=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            a = group["a"]
            b = group["b"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Decoupled (AdamW) weight decay.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Moment updates.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                denom = v_hat.sqrt_().mul_(b)          # b * sqrt(v_hat)  (v_hat consumed in place)
                update = torch.atan2(m_hat, denom)     # epsilon-free, bounded in (-pi/2, pi/2)
                p.add_(update, alpha=-lr * a)

        return loss
