"""SparseRowAdamW — per-(table, row) sparse-aware AdamW for LUT param tensors.

Each parameter is expected to be of shape `[n_tables, table_dim, n_outputs]`,
where the first two dims index a unique LUT row. Each parameter must have an
attribute `_visit_counts` of shape `[n_tables, table_dim]` (long) set by the
LUT module's forward pass BEFORE `.step()` is called; it gives the number of
tokens whose argmax selected each (table, row) this batch.

The optimizer treats c_r touches of a row this batch as equivalent to c_r
sequential single-touch Adam updates with per-touch gradient g_r/c_r:

    if c_r > 0:
        g_mean = grad_r / c_r
        m_r ← β1^{c_r} · m_r + (1 - β1^{c_r}) · g_mean
        v_r ← β2^{c_r} · v_r + (1 - β2^{c_r}) · g_mean²
        t_r += c_r
        m̂ = m_r / (1 - β1^{t_r})
        v̂ = v_r / (1 - β2^{t_r})
        θ_r -= lr * (m̂ / (√v̂ + ε) + wd · θ_r)

Untouched rows (c_r = 0) have NO state update — no decay of m_r or v_r, no
step on θ_r. This matches a "sparse Adam that has truly skipped this batch."

Non-LUT params (LayerNorm, embeddings, unembedder) should remain in a normal
torch.optim.AdamW group; this class is for LUT weights only.
"""
from __future__ import annotations
import torch


class SparseRowAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple = (0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
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
                        f"SparseRowAdamW expects 3D params [n_tables, table_dim, n_out]; "
                        f"got {p.shape}"
                    )
                c_r = getattr(p, '_visit_counts', None)
                if c_r is None:
                    raise RuntimeError(
                        f"Param of shape {tuple(p.shape)} has no _visit_counts. "
                        f"Did you register the LUT forward-hook?"
                    )
                if c_r.shape != p.shape[:2]:
                    raise RuntimeError(
                        f"visit_counts shape {tuple(c_r.shape)} != param row-shape "
                        f"{tuple(p.shape[:2])}"
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

                c_r_f = c_r.to(p.dtype)                       # [n_tables, table_dim]
                touched = c_r_f > 0                            # [n_tables, table_dim]

                # Per-row decay factors β^{c_r}. For c_r=0 they evaluate to 1.0
                # (no decay). torch.pow handles c_r=0 cleanly.
                beta1_cr = beta1 ** c_r_f                     # [n_tables, table_dim]
                beta2_cr = beta2 ** c_r_f                     # [n_tables, table_dim]

                # Reshape for broadcast over n_out dim.
                beta1_cr3 = beta1_cr.unsqueeze(-1)            # [n_tables, table_dim, 1]
                beta2_cr3 = beta2_cr.unsqueeze(-1)
                touched3  = touched.unsqueeze(-1)

                # Per-touch mean gradient. For untouched rows grad is zero
                # anyway, but clamp denominator to avoid 0/0 -> NaN.
                c_r_safe = c_r_f.clamp(min=1.0).unsqueeze(-1)
                g_mean = grad / c_r_safe

                # m ← β1^c · m + (1-β1^c) · g_mean
                m.mul_(beta1_cr3).addcmul_((1 - beta1_cr3), g_mean, value=1)
                # v ← β2^c · v + (1-β2^c) · g_mean²
                v.mul_(beta2_cr3).addcmul_((1 - beta2_cr3), g_mean * g_mean, value=1)

                # Cumulative visit count for bias correction.
                t_r.add_(c_r_f)

                # Bias-corrected estimates.
                bc1 = (1.0 - beta1 ** t_r).unsqueeze(-1).clamp(min=1e-12)
                bc2 = (1.0 - beta2 ** t_r).unsqueeze(-1).clamp(min=1e-12)
                m_hat = m / bc1
                v_hat = v / bc2

                update = m_hat / (v_hat.sqrt() + eps)

                # Decoupled weight decay (AdamW style). Apply only to touched
                # rows to preserve the "skip untouched" semantics.
                if wd != 0:
                    update = update + wd * p

                # Mask untouched rows out — no step on rows we didn't see.
                update = update * touched3.to(update.dtype)

                p.add_(update, alpha=-lr)

        return loss
