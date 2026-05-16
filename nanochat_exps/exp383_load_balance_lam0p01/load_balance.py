"""Load-balancing auxiliary loss for TinyMultiHeadLut(soft) modules.

Switch-Transformer style: for each LUT module, compute
   L_module = N · Σ_r (f_r · p_r)
where:
   N      = table_dim = 2^NAP (number of rows per table)
   f_r    = c_r / total_visits per table        (NO gradient — visit counts)
   p_r    = mean over batch of sel_soft[:, t, r] (HAS gradient — soft probs)

Loss is 1.0 when uniform, larger when concentrated. Minimising it pushes
the model to spread its soft routing across more rows. Aggregated over all
24 TinyMHLut modules, scaled by λ, added to the main NLL loss.

The hook does a SECOND soft routing computation (sign-rationalisation +
einsum + softmax) on the same input — adds ~50% to forward time at LUT
modules but well worth it for the exploration mechanism.

Memory: with bf16 autocast, peak ~ B*T × max(n_tables × table_dim) floats =
~256 MB per layer (v_lut/out_proj are largest).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut


_REGISTERED_MODULES: list[TinyMultiHeadLut] = []


def _make_load_balance_hook(mod: TinyMultiHeadLut):
    """Returns a forward_hook that computes the per-module load-balance loss
    and attaches it to mod._load_balance_loss."""

    def hook(module, inputs, output):
        x = inputs[0]
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])

        # bf16 autocast to limit memory footprint of [B*T, n_tables, 2^NAP]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=x.is_cuda):
            # Sign rationalisation (same as soft backward body)
            d = x[:, mod.soft_anchor_a_long] - x[:, mod.soft_anchor_b_long]
            T_soft = mod.log_soft_score_temp.exp()
            T_sel  = mod.log_select_temp.exp()
            p = d / (T_soft + d.abs())              # [B*T, n_tables, n_anchor_pairs]
            bit_matrix = mod.soft_bit_matrix.to(p.dtype)
            ts = torch.einsum("btp,pk->btk", p, bit_matrix)  # [B*T, n_tables, 2^NAP]
            sel_soft = F.softmax(ts / T_sel, dim=-1)
            # Per-table mean over batch → [n_tables, 2^NAP], has gradient
            p_r = sel_soft.mean(dim=0).to(torch.float32)

            # f_r from visit counts (detached, no gradient)
            c_r = mod.weights._visit_counts.to(torch.float32)
            total = c_r.sum(dim=-1, keepdim=True).clamp(min=1.0)
            f_r = (c_r / total).detach()

            # Switch-style: N · Σ_r f_r·p_r ; averaged across n_tables.
            N = p_r.shape[-1]
            module_loss = N * (f_r * p_r).sum(dim=-1).mean()

        mod._load_balance_loss = module_loss

    return hook


def install_load_balance_hooks(model: torch.nn.Module) -> int:
    """Walk model, register load-balance hooks on every soft-mode TinyMHLut.
    Returns count of modules hooked."""
    count = 0
    for mod in model.modules():
        if not isinstance(mod, TinyMultiHeadLut):
            continue
        if not hasattr(mod, 'soft_anchor_a_long'):
            continue   # only soft mode
        if id(mod) in {id(x) for x in _REGISTERED_MODULES}:
            continue
        mod.register_forward_hook(_make_load_balance_hook(mod))
        _REGISTERED_MODULES.append(mod)
        count += 1
    return count


def collect_load_balance_loss(model: torch.nn.Module) -> torch.Tensor:
    """Sum per-module load-balance losses computed by the most recent
    forward. Returns a scalar tensor (with autograd graph). If no module
    has been forwarded yet, returns 0.0 scalar."""
    total = None
    for mod in _REGISTERED_MODULES:
        l = getattr(mod, '_load_balance_loss', None)
        if l is None:
            continue
        total = l if total is None else (total + l)
    if total is None:
        # Return a zero scalar on CPU; broadcasts fine when added to main loss
        return torch.tensor(0.0)
    return total / max(1, len(_REGISTERED_MODULES))
