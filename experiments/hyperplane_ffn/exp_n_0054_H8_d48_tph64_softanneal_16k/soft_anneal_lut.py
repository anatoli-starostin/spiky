"""SoftAnnealLut — soft full-K weighted-sum forward with annealed temps AND an
adaptive handoff to the real FastMultiHeadLut hard-forward/soft-backward path.

Experiment-local (exp_n_0054); does NOT modify the shared FastMultiHeadLut /
CompressionMultiHeadLUT. It WRAPS a real per-head FastMHL (n_heads=1) and SHARES
its weight/anchor/bit_matrix tensors, so:
  * mode="soft": forward = the differentiable softmax-weighted sum over all K rows
    (pure autograd, NO straight-through estimator), temps driven by an anneal
    schedule. Grads flow straight into the wrapped FastMHL's `weights`.
  * mode="hard": forward simply delegates to the wrapped FastMHL — the standard
    hard gather forward + soft-backward STE surrogate — using the SAME weights the
    soft phase trained (no copy needed; the tensors were shared all along).

The training loop anneals the temperature soft->sharp and, once the mean top-1
softmax mass crosses a threshold (soft output ~= hard output), flips every module
to mode="hard" (the "adaptive handoff") and continues with the real STE path.

Soft math is faithful to FastMHL's `_soft_lut_bwd_body` surrogate:
    d         = x[:, anchor_a] - x[:, anchor_b]
    soft_sign = d / (T_soft + |d|)                 # -> sign(d) as T_soft -> 0
    score_k   = <bit_matrix[:,k], soft_sign>
    w         = softmax(score / T_sel, dim=K)      # -> one-hot as T_sel -> 0
    out_table = sum_k w_k * weights[table, k, :]   # bag-summed over tables_per_head
(for the argmax row FastMHL's pinned surrogate p=d/(T_soft+|d|) equals soft_sign).
"""
import torch
import torch.nn as nn


@torch.compile(dynamic=True)
def _soft_anneal_body(x, weights, anchor_a, anchor_b, bit_matrix,
                      T_soft, T_sel, n_heads: int, tph: int):
    # x: [N, input_dim]; anchor_a/b: [n_tables, nap]; weights: [n_tables, K, n_out]
    d = x[:, anchor_a] - x[:, anchor_b]                          # [N, n_tables, nap]
    soft_sign = d / (T_soft + d.abs())                          # in (-1, 1)
    score = torch.einsum("btp,pk->btk", soft_sign, bit_matrix.to(soft_sign.dtype))  # [N, n_tables, K]
    w = torch.softmax(score / T_sel, dim=-1)                    # [N, n_tables, K]
    out = torch.einsum("btk,tko->bto", w, weights)              # [N, n_tables, n_out]
    N = x.shape[0]
    n_out = weights.shape[2]
    out = out.view(N, n_heads, tph, n_out).sum(dim=2)           # [N, n_heads, n_out]
    top1 = w.amax(dim=-1).mean()                                # scalar: mean top-1 softmax mass
    return out, top1


class SoftAnnealLut(nn.Module):
    """Soft-forward + annealed-temp wrapper around a real per-head FastMultiHeadLut,
    with an adaptive handoff to the FastMHL hard path.

    `fmhl` (the wrapped FastMultiHeadLut, n_heads=1) OWNS the trainable weights and
    the routing buffers; this module holds no trainable params of its own. Call
    `set_temps(...)` each step with the annealed values; read `last_top1` for the
    handoff monitor; call `set_hard()` to perform the handoff.
    """

    def __init__(self, fmhl, temp_init: float = 0.5):
        super().__init__()
        self.fmhl = fmhl                             # real FastMHL: owns weights/anchors/temps
        self.register_buffer("T_soft", torch.tensor(float(temp_init)), persistent=False)
        self.register_buffer("T_sel", torch.tensor(float(temp_init)), persistent=False)
        self.hard = False                            # False -> soft forward; True -> delegate to fmhl
        self.last_top1 = 0.0                         # most recent mean top-1 softmax mass (detached float)

    @torch.no_grad()
    def set_temps(self, t_soft: float, t_sel: float):
        self.T_soft.fill_(float(t_soft))
        self.T_sel.fill_(float(t_sel))

    @torch.no_grad()
    def set_hard(self, seed_fmhl_temps_to: float = None):
        """Perform the handoff: switch to the real FastMHL hard path. Optionally seed
        the FastMHL's learnable log-temps to the current annealed value for continuity."""
        self.hard = True
        if seed_fmhl_temps_to is not None:
            import math
            lv = math.log(max(float(seed_fmhl_temps_to), 1e-6))
            for name in ("log_soft_score_temp", "log_select_temp"):
                p = getattr(self.fmhl, name, None)
                if p is not None:
                    p.data.fill_(lv)

    def _soft(self, x):
        return _soft_anneal_body(
            x, self.fmhl.weights, self.fmhl.soft_anchor_a_long, self.fmhl.soft_anchor_b_long,
            self.fmhl.soft_bit_matrix, self.T_soft, self.T_sel,
            int(self.fmhl.n_heads), int(self.fmhl.tables_per_head))

    def forward(self, x):
        if self.hard:
            return self.fmhl(x)                      # real hard-forward + soft-backward STE
        out, top1 = self._soft(x)
        self.last_top1 = float(top1.detach())
        return out

    @torch.no_grad()
    def soft_hard_gap(self, x):
        """Relative L2 gap ||soft(x) - hard(x)|| / ||hard(x)|| on the current weights."""
        soft, _ = self._soft(x)
        hard = self.fmhl(x)
        num = (soft - hard).pow(2).sum().sqrt()
        den = hard.pow(2).sum().sqrt() + 1e-8
        return float((num / den).item())


def anneal_temp(step: int, n_steps: int, start: float, floor: float) -> float:
    """Exponential decay start -> floor over n_steps (frac clamped to [0,1])."""
    frac = min(max(step, 0) / max(n_steps, 1), 1.0)
    return float(start) * (float(floor) / float(start)) ** frac
