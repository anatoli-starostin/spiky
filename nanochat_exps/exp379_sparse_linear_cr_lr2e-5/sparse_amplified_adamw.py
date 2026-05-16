"""SparseAmplifiedAdamW — sparse-aware AdamW that lets hot rows step further.

Supports three modes via the `mode` arg:

  - "step_sqrt_cr"   (A):  θ_r -= lr · √c_r · m̂_r / (√v̂_r + ε)
  - "step_linear_cr" (B):  θ_r -= lr · c_r   · m̂_r / (√v̂_r + ε)
  - "ema_weight_cr"  (C):  α = c_r / (c_r + κ);
                            m_r = (1-α)·m_r + α·g_r;  v_r = (1-α)·v_r + α·g_r²
                            θ_r -= lr · m̂_r / (√v̂_r + ε)

Modes A and B keep vanilla EMA mass (β1=0.9, β2=0.95). Mode C overrides EMA
mass with a Bayesian-flavored c_r-dependent weight using `kappa` as prior.

In all modes:
  - m_r, v_r do NOT decay on untouched rows (c_r == 0).
  - t_r counts batches where row r was touched (per-row bias correction).
  - No parameter update on untouched rows.

Each parameter must have `_visit_counts` (shape [n_tables, table_dim], long)
set by the LUT forward hook prior to .step().
"""
from __future__ import annotations
import torch


class SparseAmplifiedAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple = (0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 mode: str = "step_sqrt_cr", kappa: float = 8.0):
        if mode not in ("step_sqrt_cr", "step_linear_cr", "ema_weight_cr"):
            raise ValueError(f"Unknown mode: {mode}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        mode=mode, kappa=kappa)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            mode = group['mode']
            kappa = group['kappa']

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 3:
                    raise RuntimeError(
                        f"SparseAmplifiedAdamW expects 3D params [n_tables, table_dim, n_out]; "
                        f"got {p.shape}"
                    )
                c_r = getattr(p, '_visit_counts', None)
                if c_r is None:
                    raise RuntimeError(
                        f"Param of shape {tuple(p.shape)} has no _visit_counts."
                    )

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['m']   = torch.zeros_like(p)
                    state['v']   = torch.zeros_like(p)
                    state['t_r'] = torch.zeros(p.shape[:2], device=p.device, dtype=torch.float32)

                m = state['m']
                v = state['v']
                t_r = state['t_r']

                touched = (c_r > 0)
                touched3 = touched.unsqueeze(-1)
                c_r_f = c_r.to(p.dtype)

                if mode == "ema_weight_cr":
                    # Per-row α = c_r / (c_r + κ); cap untouched at 0.
                    alpha = (c_r_f / (c_r_f + kappa)).unsqueeze(-1)
                    one_minus_alpha = (1.0 - alpha)
                    new_m = one_minus_alpha * m + alpha * grad
                    new_v = one_minus_alpha * v + alpha * (grad * grad)
                else:
                    # Modes A and B: vanilla EMA on touched rows.
                    new_m = beta1 * m + (1 - beta1) * grad
                    new_v = beta2 * v + (1 - beta2) * grad * grad

                m.copy_(torch.where(touched3, new_m, m))
                v.copy_(torch.where(touched3, new_v, v))

                # Per-row batch-touch counter.
                t_r.add_(touched.to(t_r.dtype))

                # Bias correction.
                if mode == "ema_weight_cr":
                    # Note: with non-uniform EMA weight, bias correction is
                    # less clean; approximate using t_r and avg α
                    # but for simplicity we just use β1, β2 cumulative products.
                    bc1 = (1.0 - beta1 ** t_r).unsqueeze(-1).clamp(min=1e-12)
                    bc2 = (1.0 - beta2 ** t_r).unsqueeze(-1).clamp(min=1e-12)
                else:
                    bc1 = (1.0 - beta1 ** t_r).unsqueeze(-1).clamp(min=1e-12)
                    bc2 = (1.0 - beta2 ** t_r).unsqueeze(-1).clamp(min=1e-12)
                m_hat = m / bc1
                v_hat = v / bc2

                update = m_hat / (v_hat.sqrt() + eps)

                # Step amplifier.
                if mode == "step_sqrt_cr":
                    amp = c_r_f.sqrt().unsqueeze(-1)
                elif mode == "step_linear_cr":
                    amp = c_r_f.unsqueeze(-1)
                else:
                    amp = touched3.to(update.dtype)  # 1.0 on touched, 0 elsewhere

                update = update * amp

                if wd != 0:
                    update = update + wd * p * touched3.to(p.dtype)

                p.add_(update, alpha=-lr)

        return loss
