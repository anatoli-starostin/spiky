"""Forward hook on TinyMultiHeadLut modules: attaches per-(table, row)
visit counts to `mod.weights._visit_counts` after every forward pass.

We re-derive the per-token argmax row index from x using the same
`_soft_index_signpack` logic the module's forward uses internally (sign-bit
pack of pair distances). For configs with `argmax_noise_eps>0` this would
disagree from the in-kernel index due to bernoulli randomness; current
configs use eps=0 so the re-computed index matches exactly.

Then we bincount the per-(token, table) indices along the token dim to
get visit counts of shape [n_tables, table_dim] (long), attached to the
weights parameter so SparseRowAdamW can read it at step time.

Only soft-mode TinyMultiHeadLut modules are hooked. Multi-alt modules
are not currently used in the bs=16 LUT-LM regime; they would need a
separate hook.
"""
from __future__ import annotations
import torch
from spiky.lutorch.tiny_multi_head_lut import (
    TinyMultiHeadLut, _soft_index_signpack,
)


_HOOKED_MODULES = set()


def _make_hook(mod: TinyMultiHeadLut):
    n_anchor_pairs = mod.lookup.anchor_pairs_a.shape[-1]
    n_tables = mod.weights.shape[0]
    table_dim = mod.weights.shape[1]
    noise_eps = float(getattr(mod, 'argmax_noise_eps', 0.0))
    anchor_a = mod.soft_anchor_a_long
    anchor_b = mod.soft_anchor_b_long
    powers = mod.soft_powers

    @torch.no_grad()
    def hook(module, inputs, output):
        x = inputs[0]
        # x shape: [B*T, E] for our usage (model flattens before calling).
        # _soft_index_signpack expects [B, E] and uses anchor indexing into
        # the last dim; same shape works for [B*T, E].
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])
        index = _soft_index_signpack(x, anchor_a, anchor_b, powers, noise_eps)
        # index shape: [B*T, n_tables]
        idx_t = index.t().contiguous().to(torch.int64)
        counts = torch.zeros(n_tables, table_dim,
                             device=index.device, dtype=torch.int64)
        ones = torch.ones_like(idx_t, dtype=torch.int64)
        counts.scatter_add_(1, idx_t, ones)
        # Attach to the Parameter — SparseRowAdamW reads `_visit_counts`.
        mod.weights._visit_counts = counts

    return hook


def install_visit_trackers(model: torch.nn.Module) -> int:
    """Walk model, find TinyMultiHeadLut(soft) modules, register hooks. Returns count."""
    count = 0
    for mod in model.modules():
        if not isinstance(mod, TinyMultiHeadLut):
            continue
        if id(mod) in _HOOKED_MODULES:
            continue
        # Only soft mode — multi-alt path doesn't use _soft_index_signpack.
        if not hasattr(mod, 'soft_anchor_a_long'):
            continue
        mod.register_forward_hook(_make_hook(mod))
        _HOOKED_MODULES.add(id(mod))
        count += 1
    return count
