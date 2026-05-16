"""SparseSkipAdamW — minimal sparse-aware AdamW for LUT param tensors.

Same as vanilla AdamW EXCEPT:
1. m_r and v_r do NOT decay on steps where row r is untouched (c_r == 0).
2. Bias correction is per-row, using t_r = number of batches where row r
   was touched (not the global step count).
3. No parameter update at all on untouched rows (no wd step either).

This is essentially `torch.optim.SparseAdam` semantics, adapted for our
LUT param shape [n_tables, table_dim, n_outputs] where each (table, row)
in the first two dims has its own sparse update pattern.

Compared to SparseRowAdamW (exp375), this drops the β^{c_r} per-touch
weighting and the g_r/c_r normalisation — they were over-correcting. We
keep only the actual fix: don't penalise sparse rows with no-op decay.

Each parameter must have a `_visit_counts` attribute of shape
[n_tables, table_dim] (long) set by the LUT forward hook.
"""
from __future__ import annotations
import torch


class SparseSkipAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple = (0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 3:
                    raise RuntimeError(
                        f"SparseSkipAdamW expects 3D params [n_tables, table_dim, n_out]; "
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

                touched = (c_r > 0)                            # [n_tables, table_dim]
                touched3 = touched.unsqueeze(-1)                # broadcastable

                # Standard EMA but conditional: only update m,v where touched.
                # Compute the candidate new values, then mix using `touched` mask.
                new_m = beta1 * m + (1 - beta1) * grad
                new_v = beta2 * v + (1 - beta2) * grad * grad
                m.copy_(torch.where(touched3, new_m, m))
                v.copy_(torch.where(touched3, new_v, v))

                # Per-row batch-touch count (increment by 1 for touched rows only).
                t_r.add_(touched.to(t_r.dtype))

                # Bias-corrected estimates with per-row t_r.
                bc1 = (1.0 - beta1 ** t_r).unsqueeze(-1).clamp(min=1e-12)
                bc2 = (1.0 - beta2 ** t_r).unsqueeze(-1).clamp(min=1e-12)
                m_hat = m / bc1
                v_hat = v / bc2

                update = m_hat / (v_hat.sqrt() + eps)
                if wd != 0:
                    update = update + wd * p

                # Mask untouched rows to zero step.
                update = update * touched3.to(update.dtype)

                p.add_(update, alpha=-lr)

        return loss
