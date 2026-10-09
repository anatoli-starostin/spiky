"""Narrow-table read for the ConfidenceLUT family: gather table rows from a bf16 / fp8 COPY of the fp32 master.

The fp32 master ``weights`` stays the trained parameter (the optimizer only ever sees fp32). A narrow copy is cast
from it and gathered from instead, so the read moves 2x (bf16) / 4x (fp8) fewer bytes per row; the score-weighted
sum and every gradient are accumulated in fp32.

* :class:`NarrowTableCache` casts ONCE per change of the master, not once per forward, so with 32 micro-batches
  per optimizer step the cast is amortised 32x. Invalidation has two layers, because the version counter alone is
  NOT reliable (measured, torch 2.9.1: ``torch.optim.AdamW/Adam(fused=True)`` and any ``param.data`` write update
  the values WITHOUT bumping ``_version``; SGD, foreach AdamW, ``copy_`` and some custom optimizers (e.g. fused or
  Muon-style wrappers that ``copy_`` into the parameter) do bump it):
    1. a global ``torch.optim`` step post-hook invalidates every live cache after ANY optimizer's ``step()``
       (fused included) - the recast then happens on the next forward, i.e. once per step;
    2. the key also holds the master's version counter + storage pointer + dtype, catching in-place updates made
       outside an optimizer (``copy_``, ``load_state_dict``).
  Writes through ``param.data`` outside an optimizer step are invisible to both: call :meth:`invalidate` (or
  :func:`invalidate_narrow_tables`) after them.
* fp8 (``float8_e4m3fn``) uses one tensorwise amax scale (the copy holds ``W * 448/amax``); the read multiplies
  the fp32 sum by the inverse scale.
* :class:`_NarrowScoredRead` (autograd): forward ``y[b] = sum_t s[b,t] * W_n[idx[b,t]]`` in fp32 from the narrow
  rows; backward gives ``d s`` from a re-gather of the NARROW rows (``(W_n[idx] * go).sum(-1)``) and the table
  gradient straight to the fp32 master via ``aten._embedding_bag_dense_backward`` (the same op embedding_bag's
  backward uses, fp32 in / fp32 out). The narrow copy never receives a gradient.

``embedding_bag`` itself cannot be used: with a bf16 table it demands bf16 per-sample weights and returns bf16
(rounding the scores and the sum), and with an fp8 table it is not implemented at all. The gather + upcast +
weighted sum written here is fused into one kernel by Inductor when the cartridge forward is compiled (CUDA).
"""
from __future__ import annotations

import weakref
from typing import Optional

import torch

# None = the default embedding_bag read. float32 = the same fused gather read as the narrow forms but straight from
# the fp32 master (no copy): the control that separates "fused read instead of embedding_bag" from "fewer bytes".
NARROW_TABLE_DTYPES = (None, torch.float32, torch.bfloat16, torch.float8_e4m3fn)
_E4M3_MAX = 448.0

_LIVE_CACHES: "weakref.WeakSet[NarrowTableCache]" = weakref.WeakSet()
_STEP_HOOK = None


def invalidate_narrow_tables(module: Optional[torch.nn.Module] = None) -> None:
    """Invalidate the narrow-table caches of ``module`` (every live cache if None); the next forward recasts."""
    caches = list(_LIVE_CACHES) if module is None else [
        m._narrow_cache for m in module.modules() if isinstance(getattr(m, "_narrow_cache", None), NarrowTableCache)]
    for c in caches:
        c.invalidate()


def _ensure_step_hook() -> None:
    global _STEP_HOOK
    if _STEP_HOOK is None:
        from torch.optim.optimizer import register_optimizer_step_post_hook
        _STEP_HOOK = register_optimizer_step_post_hook(lambda optimizer, args, kwargs: invalidate_narrow_tables())


class NarrowTableCache:
    """fp32 master -> narrow copy, recast only when the master changed (see module docstring)."""

    def __init__(self):
        self.key = None
        self.table: Optional[torch.Tensor] = None
        self.inv_scale: Optional[torch.Tensor] = None
        self.n_casts = 0
        _LIVE_CACHES.add(self)
        _ensure_step_hook()

    def invalidate(self) -> None:
        self.key = None

    @torch.no_grad()
    def get(self, master: torch.Tensor, dtype: torch.dtype):
        if dtype == master.dtype:                                  # no narrowing: read the master itself, no copy
            if self.inv_scale is None or self.inv_scale.device != master.device:
                self.inv_scale = torch.ones((), device=master.device, dtype=torch.float32)
            return master.detach(), self.inv_scale
        key = (master._version, master.data_ptr(), dtype, master.device, tuple(master.shape))
        if key != self.key:
            W = master.detach()
            if dtype == torch.float8_e4m3fn:
                scale = _E4M3_MAX / W.abs().max().float().clamp_min(1e-30)
                self.table = (W.float() * scale).to(dtype)
                self.inv_scale = scale.reciprocal()
            else:
                self.table = W.to(dtype)
                self.inv_scale = torch.ones((), device=W.device, dtype=torch.float32)
            self.key = key
            self.n_casts += 1
        return self.table, self.inv_scale


def _rows(narrow: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``narrow[idx]`` upcast to fp32. CPU has no fp8 indexing kernel ("index_cpu" not implemented for
    Float8_e4m3fn), so there the same bytes are gathered through a uint8 view and reinterpreted."""
    if narrow.device.type == "cpu" and narrow.dtype == torch.float8_e4m3fn:
        return narrow.view(torch.uint8)[idx].view(narrow.dtype).float()
    return narrow[idx].float()


class _NarrowScoredRead(torch.autograd.Function):
    """y[b] = inv_scale * sum_t s[b,t] * W_n[idx[b,t]] (fp32); grads to s (narrow re-gather) and to the fp32 master."""

    @staticmethod
    def forward(ctx, idx, s, master, narrow, inv_scale):
        ctx.save_for_backward(idx, s, narrow, inv_scale)
        ctx.num_weights = master.shape[0]
        return (_rows(narrow, idx) * s.unsqueeze(-1)).sum(1) * inv_scale

    @staticmethod
    def backward(ctx, go):
        idx, s, narrow, inv_scale = ctx.saved_tensors
        go = go.contiguous()
        grad_s = (_rows(narrow, idx) * go.unsqueeze(1)).sum(-1) * inv_scale if ctx.needs_input_grad[1] else None
        grad_w = None
        if ctx.needs_input_grad[2]:
            n_bags, n_per = idx.shape
            flat = idx.reshape(-1)
            offset2bag = torch.arange(n_bags, device=idx.device, dtype=flat.dtype).repeat_interleave(n_per)
            bag_size = torch.full((n_bags,), n_per, device=idx.device, dtype=flat.dtype)
            no_max = torch.empty(0, device=idx.device, dtype=flat.dtype)
            grad_w = torch.ops.aten._embedding_bag_dense_backward(
                go, flat, offset2bag, bag_size, no_max, ctx.num_weights, False, 0, s.reshape(-1).to(go.dtype), -1)
        return None, grad_s, grad_w, None, None


def narrow_scored_read(idx: torch.Tensor, s: torch.Tensor, master: torch.Tensor, narrow: torch.Tensor,
                       inv_scale: torch.Tensor) -> torch.Tensor:
    """``[n_bags, n_per]`` flat row indices + per-row weights -> ``[n_bags, d_out]`` fp32 (see module docstring)."""
    return _NarrowScoredRead.apply(idx, s, master, narrow, inv_scale)
