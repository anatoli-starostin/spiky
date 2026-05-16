"""Plain Gaussian noise on TinyMultiHeadLut inputs (Option α).

Forward pre-hook: during training, replaces x with x + σ·randn_like(x).
Only fires when module.training is True (no noise at eval).

The visit_tracker hook (post-forward) will see the noisy x and recompute
the index from that same noisy x — consistent with what the LUT forward
actually used.
"""
from __future__ import annotations
import torch
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut


_REGISTERED_MODULES = set()


def _make_noise_pre_hook(sigma: float):
    def hook(module, inputs):
        if not module.training:
            return None
        x = inputs[0]
        x_noisy = x + sigma * torch.randn_like(x)
        return (x_noisy,) + inputs[1:]
    return hook


def install_input_noise_hooks(model: torch.nn.Module, sigma: float) -> int:
    """Register Gaussian-noise pre-hook on every TinyMultiHeadLut(soft).
    Returns count installed."""
    if sigma <= 0:
        return 0
    count = 0
    for mod in model.modules():
        if not isinstance(mod, TinyMultiHeadLut):
            continue
        if not hasattr(mod, 'soft_anchor_a_long'):
            continue
        if id(mod) in _REGISTERED_MODULES:
            continue
        mod.register_forward_pre_hook(_make_noise_pre_hook(sigma))
        _REGISTERED_MODULES.add(id(mod))
        count += 1
    return count
