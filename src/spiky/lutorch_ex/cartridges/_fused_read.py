"""Fused read for the ConfidenceLUT family (``ConfidenceLUT(fused_read=True)``).

The default training read is one ``F.embedding_bag`` with per-sample weights (the scores). Its backward is two
PyTorch kernels: the table gradient (``_embedding_bag_dense_backward``) and the score gradient
(``_embedding_bag_per_sample_weights_backward``), and at this shape the score-gradient kernel dominates the read.

:class:`_FusedScoredRead` (autograd) replaces all three kernels, reading straight from the fp32 master table (the
trained parameter; no copy):

* forward: ``y[b] = sum_t s[b,t] * W[idx[b,t]]`` - gather, x score, sum;
* backward: ``d s`` from a re-gather of the same rows (``(W[idx] * go).sum(-1)``), and the table gradient
  ``dW[r] = sum_{(b,t): idx[b,t] = r} s[b,t] * go[b]`` accumulated with ``index_add_`` (see :data:`TABLE_GRAD`).

Fusion is conditional: the gather + weighted sum, the score-gradient re-gather and the weighted-row scatter of the
table gradient become a few Inductor kernels only when the cartridge forward is compiled (CUDA, the default). Eager
runs them as separate ops and MATERIALISES the gathered / weighted ``[n_bags * n_per, d_out]`` rows - a large
intermediate at training batch, warned about once (:func:`_warn_eager_intermediate`).

Table-gradient modes (``LUTORCH_EX_TABLE_GRAD``, read once at import; :data:`TABLE_GRAD` is the switch):

* ``index_add`` (default): ``zeros(rows, d_out).index_add_(0, idx, s * go)``. Atomic fp32 adds, so the result agrees
  with the sorted embedding_bag reduction to fp32 re-association but is not bit-reproducible run to run (the CUDA
  training step already is not: other atomic scatters in it). Not bound by embedding_bag's 32-bit launch limit.
* ``scatter_add``: the same through ``scatter_add_`` (measured identical speed); selectable, not the default.
* ``embedding_bag``: ``aten._embedding_bag_dense_backward`` - the op embedding_bag's own backward uses; bit-identical
  to the default read's table gradient, but subject to its size limit (README: "PyTorch's embedding_bag backward size
  limit"). The escape hatch.
"""
from __future__ import annotations

import os
import warnings

import torch

TABLE_GRAD_MODES = ("index_add", "scatter_add", "embedding_bag")
TABLE_GRAD = os.environ.get("LUTORCH_EX_TABLE_GRAD", "index_add")
if TABLE_GRAD not in TABLE_GRAD_MODES:
    raise ValueError(f"LUTORCH_EX_TABLE_GRAD must be one of {TABLE_GRAD_MODES}, got {TABLE_GRAD!r}")

# Eager CUDA warning threshold for the materialised [n_bags * n_per, d_out] rows (GiB).
EAGER_WARN_GIB = float(os.environ.get("LUTORCH_EX_EAGER_ROWS_WARN_GIB", "1"))
_EAGER_WARNED = False


def _warn_eager_intermediate(n_rows: int, d_out: int, dtype: torch.dtype, device: torch.device) -> None:
    """Warn ONCE when an eager CUDA backward is about to materialise the gathered/weighted rows above the threshold.
    Called only outside torch.compile (the caller checks torch.compiler.is_compiling(), which Dynamo constant-folds, so
    the compiled graph never contains this): it fires for real eager execution - e.g. LUTORCH_EX_NO_COMPILE=1 - where
    nothing fuses the rows away."""
    global _EAGER_WARNED
    gib = n_rows * d_out * torch.empty((), dtype=dtype).element_size() / 2 ** 30
    if _EAGER_WARNED or device.type != "cuda" or gib <= EAGER_WARN_GIB:
        return
    _EAGER_WARNED = True
    warnings.warn(
        f"lutorch_ex fused read running EAGER on CUDA: its backward materialises a [{n_rows:,}, {d_out}] "
        f"{str(dtype).replace('torch.', '')} intermediate ({gib:.1f} GiB, plus the same again for the score gradient's "
        f"re-gather). Compiled (the default on CUDA) these are fused away. Compile the cartridge, reduce the "
        f"micro-batch, or raise LUTORCH_EX_EAGER_ROWS_WARN_GIB to silence this.", RuntimeWarning, stacklevel=3)


def _table_grad(idx: torch.Tensor, s: torch.Tensor, go: torch.Tensor, num_weights: int) -> torch.Tensor:
    """``[num_weights, d_out]`` table gradient for idx ``[n_bags, n_per]``, s ``[n_bags, n_per]``, go
    ``[n_bags, d_out]``, accumulated per :data:`TABLE_GRAD`."""
    n_bags, n_per = idx.shape
    flat = idx.reshape(-1)
    if TABLE_GRAD == "embedding_bag":
        offset2bag = torch.arange(n_bags, device=idx.device, dtype=flat.dtype).repeat_interleave(n_per)
        bag_size = torch.full((n_bags,), n_per, device=idx.device, dtype=flat.dtype)
        no_max = torch.empty(0, device=idx.device, dtype=flat.dtype)
        return torch.ops.aten._embedding_bag_dense_backward(
            go, flat, offset2bag, bag_size, no_max, num_weights, False, 0, s.reshape(-1).to(go.dtype), -1)
    d_out = go.shape[-1]
    src = (s.to(go.dtype).unsqueeze(-1) * go.unsqueeze(1)).reshape(-1, d_out)       # [n_bags * n_per, d_out]
    grad_w = go.new_zeros(num_weights, d_out)
    if TABLE_GRAD == "index_add":
        return grad_w.index_add_(0, flat, src)
    return grad_w.scatter_add_(0, flat.to(torch.int64).unsqueeze(-1).expand(-1, d_out), src)


class _FusedScoredRead(torch.autograd.Function):
    """y[b] = sum_t s[b,t] * W[idx[b,t]]; grads to s (re-gather) and to the table W (:func:`_table_grad`)."""

    @staticmethod
    def forward(ctx, idx, s, weights):
        ctx.save_for_backward(idx, s, weights)
        return (weights[idx] * s.unsqueeze(-1)).sum(1)

    @staticmethod
    def backward(ctx, go):
        idx, s, weights = ctx.saved_tensors
        go = go.contiguous()
        if not torch.compiler.is_compiling():       # constant-folded by Dynamo: the traced graph never contains this
            _warn_eager_intermediate(idx.numel(), go.shape[-1], go.dtype, go.device)
        grad_s = (weights[idx] * go.unsqueeze(1)).sum(-1) if ctx.needs_input_grad[1] else None
        grad_w = _table_grad(idx, s, go, weights.shape[0]) if ctx.needs_input_grad[2] else None
        return None, grad_s, grad_w


def fused_scored_read(idx: torch.Tensor, s: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``[n_bags, n_per]`` flat row indices + per-row scores, ``[rows, d_out]`` table -> ``[n_bags, d_out]``."""
    return _FusedScoredRead.apply(idx, s, weights)
