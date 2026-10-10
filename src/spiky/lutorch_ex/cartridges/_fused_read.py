"""Fused read for the ConfidenceLUT family (``ConfidenceLUT(fused_read=True)``).

The default training read is one ``F.embedding_bag`` with per-sample weights (the scores). Its backward is two
PyTorch kernels: the table gradient (``_embedding_bag_dense_backward``) and the score gradient
(``_embedding_bag_per_sample_weights_backward``), and at this shape the score-gradient kernel dominates the read.

:class:`_FusedScoredRead` (autograd) replaces embedding_bag's FORWARD and SCORE-GRADIENT kernels, reading straight
from the fp32 master table (the trained parameter; no copy):

* forward: ``y[b] = sum_t s[b,t] * W[idx[b,t]]`` - gather, x score, sum;
* backward: ``d s`` from a re-gather of the same rows (``(W[idx] * go).sum(-1)``), and the table gradient through
  ``aten._embedding_bag_dense_backward`` - the same op embedding_bag's own backward uses - so it keeps that op's size
  limit (README: "PyTorch's embedding_bag backward size limit").

Fusion is conditional: the gather + weighted sum (and the score-gradient re-gather) become single kernels only when
Inductor compiles the cartridge forward (CUDA); eager runs them as separate ops and materialises the gathered rows.
"""
from __future__ import annotations

import torch


class _FusedScoredRead(torch.autograd.Function):
    """y[b] = sum_t s[b,t] * W[idx[b,t]]; grads to s (re-gather) and to the table W (embedding_bag's dense backward)."""

    @staticmethod
    def forward(ctx, idx, s, weights):
        ctx.save_for_backward(idx, s, weights)
        return (weights[idx] * s.unsqueeze(-1)).sum(1)

    @staticmethod
    def backward(ctx, go):
        idx, s, weights = ctx.saved_tensors
        go = go.contiguous()
        grad_s = (weights[idx] * go.unsqueeze(1)).sum(-1) if ctx.needs_input_grad[1] else None
        grad_w = None
        if ctx.needs_input_grad[2]:
            n_bags, n_per = idx.shape
            flat = idx.reshape(-1)
            offset2bag = torch.arange(n_bags, device=idx.device, dtype=flat.dtype).repeat_interleave(n_per)
            bag_size = torch.full((n_bags,), n_per, device=idx.device, dtype=flat.dtype)
            no_max = torch.empty(0, device=idx.device, dtype=flat.dtype)
            grad_w = torch.ops.aten._embedding_bag_dense_backward(
                go, flat, offset2bag, bag_size, no_max, weights.shape[0], False, 0, s.reshape(-1).to(go.dtype), -1)
        return None, grad_s, grad_w


def fused_scored_read(idx: torch.Tensor, s: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``[n_bags, n_per]`` flat row indices + per-row scores, ``[rows, d_out]`` table -> ``[n_bags, d_out]``."""
    return _FusedScoredRead.apply(idx, s, weights)
