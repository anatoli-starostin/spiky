"""The ConfidenceLUT read in its compiled form: gather × weight → sum with its own backward.

``F.embedding_bag``'s backward is two PyTorch kernels, the table gradient (``_embedding_bag_dense_backward``) and the
weight gradient (``_embedding_bag_per_sample_weights_backward``); at the champion shape the weight-gradient kernel
dominates the read. :class:`_FusedScoredRead` replaces all three kernels, reading straight from the float32 table (the
trained parameter; no copy):

* forward: ``y[b] = sum_k w[b,k] * W[rows[b,k]]``, a gather, × weight, sum;
* backward: ``d w`` from a re-gather of the same rows (``(W[rows] * go).sum(-1)``), and the table gradient
  ``dW[r] = sum_{(b,k): rows[b,k] = r} w[b,k] * go[b]`` accumulated with ``index_add_``.

ConfidenceLUT calls this only inside ``torch.compile``, where Inductor fuses the gather and weighted sum, the
re-gather and the weighted-row scatter into a few kernels. Uncompiled, those ops would materialise the gathered rows,
so the eager form of the read is ``F.embedding_bag`` (see ``confidence.py``).

The ``index_add_`` uses atomic float32 adds: the table gradient agrees with embedding_bag's sorted reduction to float32
re-association, not bit for bit, and is not bit-reproducible run to run (the CUDA training step already is not: it has
other atomic scatters). It is not bound by embedding_bag's 32-bit launch limit (_fused_ops.py).
"""
from __future__ import annotations

import torch


class _FusedScoredRead(torch.autograd.Function):
    """y[b] = sum_k w[b,k] * W[rows[b,k]]; gradients to w (re-gather) and to the table W (index_add_)."""

    @staticmethod
    def forward(ctx, rows, w, table):
        ctx.save_for_backward(rows, w, table)
        return (table[rows] * w.unsqueeze(-1)).sum(1)

    @staticmethod
    def backward(ctx, go):
        rows, w, table = ctx.saved_tensors
        go = go.contiguous()
        grad_w = (table[rows] * go.unsqueeze(1)).sum(-1) if ctx.needs_input_grad[1] else None
        grad_table = None
        if ctx.needs_input_grad[2]:
            src = (w.to(go.dtype).unsqueeze(-1) * go.unsqueeze(1)).reshape(-1, go.shape[-1])
            grad_table = go.new_zeros(table.shape).index_add_(0, rows.reshape(-1), src)
        return None, grad_w, grad_table


def fused_scored_read(rows: torch.Tensor, w: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``[n_bags, n_per]`` flat row indices and their weights, ``[n_rows, d_out]`` table -> ``[n_bags, d_out]``."""
    return _FusedScoredRead.apply(rows, w, table)
