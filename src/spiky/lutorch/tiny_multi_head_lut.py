"""TinyMultiHeadLut — minimal MultiHeadLut variant for fast training/inference.

Sibling of MultiHeadLut, mirroring TinyAnchorPairsLookup's "stripped-down for
the common case" style. Hardcoded simplifications:

  - smooth_mode = False (no per-alternative soft routing)
  - n_alternatives = 1 (single carrier, hard lookup)
  - n_anchor_pairs <= 15 (TinyAnchorPairsLookup int16 constraint)
  - input_dim <= 32767 (int16 anchor index limit)
  - cmp_eps = 0
  - anchor_sampling_policy ∈ {CANONICAL_FULL_COVERAGE, CANONICAL_DISTINCT}
  - shuffle_per_head = True
  - n_buckets = 1 (no bucket conditioning)

Design choices vs MultiHeadLut:
  - Weights stored in user-chosen dtype (default torch.bfloat16) for ~2× memory
    savings vs fp32 MultiHeadLut. Forward gather, backward scatter both work
    natively in bf16/fp16 on H100+ (Tensor Core paths).
  - No per-table-output materialisation: forward computes the [B, n_lookup_tables,
    n_outputs] gather, immediately reduces to [B, n_heads, n_outputs] by summing
    over tables_per_head. Backward through advanced indexing handled by autograd.
  - Custom Adam optimiser (TinyMultiHeadLutOptimizer) keeps m, v in the same
    weight dtype (bf16 by default) for matching memory savings.

Forward signature (matches MultiHeadLut's reduced output):
  x: float [B, input_dim]
  returns: [B, n_heads, n_outputs] in weights' dtype.
"""
from typing import Optional

import torch
import torch.nn as nn

from spiky.lutorch.tiny_anchor_pairs_lookup import TinyAnchorPairsLookup
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

# Toggle to disable the native fused backward kernel (PyTorch fallback path).
_USE_TINY_MHLUT_NATIVE_BWD = True

_NATIVE_MHLUT = None
def _get_tiny_mhlut_native():
    """Lazily fetch the native LUTorchManager and check for the fused-bwd binding."""
    global _NATIVE_MHLUT
    if _NATIVE_MHLUT is not None:
        return _NATIVE_MHLUT
    try:
        import lutorch_cuda  # noqa: F401  (loaded for side effects)
        m = lutorch_cuda.get_lutorch_manager()
        if hasattr(m, 'tiny_mhlut_backward_na1'):
            _NATIVE_MHLUT = m
            return m
    except Exception:
        pass
    return None


def _embedding_bag_forward(weights: torch.Tensor, lookup_indices: torch.Tensor,
                           n_heads: int, tables_per_head: int) -> torch.Tensor:
    """Fused gather + reduce via F.embedding_bag (mode='sum'). Used by both
    the training autograd Function and the eval no-grad shortcut."""
    B, n_lookup_tables = lookup_indices.shape
    table_dim = weights.shape[1]
    n_outputs = weights.shape[2]
    weights_flat = weights.view(n_lookup_tables * table_dim, n_outputs)
    table_offset = (
        torch.arange(n_lookup_tables, device=weights.device, dtype=lookup_indices.dtype)
        * table_dim
    )
    flat_indices = (lookup_indices + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tables_per_head
    out_flat = torch.nn.functional.embedding_bag(
        flat_indices, weights_flat, offsets=offsets, mode='sum',
    )
    return out_flat.view(B, n_heads, n_outputs)


class _TinyMHLutGatherReduce(torch.autograd.Function):
    """Gather + reduce + STE-carrier-thread, with recompute-in-backward.

    Forward:
        weights: [n_lookup_tables, table_dim, n_outputs]
        lookup_indices: [B, n_lookup_tables] int64 (chosen anchor pair, "main")
        lookup_alt_indices: [B, n_lookup_tables] int64 (runner-up, "alt")
        lookup_indices_grad_c: [B, n_lookup_tables] zero-valued main carrier
            with autograd link to x (via TinyAnchorPairsLookup).
        lookup_alt_indices_grad_c: [B, n_lookup_tables] zero-valued alt carrier
            with the matching autograd link.
        n_heads, tables_per_head: shape ints (saved on ctx).

    Both carriers are required for x.grad correctness: TinyAnchorPairsLookup's
    backward computes the STE update as
        du = (grad_main - grad_alt) * uncertainty_derivative,
    matching MHLut's `lprojection_backward_na1_carriers_kernel`. Threading
    only the main carrier (i.e. setting grad_alt=0) silently breaks
    numerical equivalence with MultiHeadLut and causes a structural ~+0.03
    bpb gap in downstream training.

    Saves only `weights`, `lookup_indices`, `lookup_alt_indices` (the
    [B, n_lookup_tables, n_outputs] gather is NOT saved — recomputed in
    backward).
    """

    @staticmethod
    def forward(ctx, weights, lookup_indices, lookup_alt_indices,
                lookup_indices_grad_c, lookup_alt_indices_grad_c,
                n_heads: int, tables_per_head: int):
        out = _embedding_bag_forward(weights, lookup_indices, n_heads, tables_per_head)
        # Save weights (parameter — no extra mem) and both lookup index
        # tensors (small int64). The big gather is never materialised.
        ctx.save_for_backward(weights, lookup_indices, lookup_alt_indices)
        ctx.n_heads = n_heads
        ctx.tables_per_head = tables_per_head
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [B, n_heads, n_outputs] in weights' dtype (or upstream's).
        weights, lookup_indices, lookup_alt_indices = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tables_per_head
        B, n_lookup_tables = lookup_indices.shape
        n_outputs = weights.shape[2]
        table_dim = weights.shape[1]

        if grad_out.dtype != weights.dtype:
            grad_out = grad_out.to(weights.dtype)

        # Native fused path: weights + carriers (main+alt) kernels modeled on
        # MHLut's lprojection_backward_na1_*. Returns grad_main AND grad_alt
        # so TinyAnchorPairsLookup's bwd can compute du = (grad_main-grad_alt)
        # * uncertainty_derivative for the STE update on x.grad.
        native = _get_tiny_mhlut_native() if _USE_TINY_MHLUT_NATIVE_BWD else None
        if native is not None and weights.is_cuda:
            grad_weights, grad_main, grad_alt = native.tiny_mhlut_backward_na1(
                grad_out.contiguous(),
                weights,
                lookup_indices.contiguous(),
                lookup_alt_indices.contiguous(),
                tph,
            )
            return grad_weights, None, None, grad_main, grad_alt, None, None

        # PyTorch fallback path. Mirror native: compute grad_main and grad_alt
        # by gathering at main and alt indices respectively.
        table_ix = torch.arange(n_lookup_tables, device=weights.device).view(1, -1).expand(B, -1)
        out_main = weights[table_ix, lookup_indices]      # [B, n_lookup_tables, n_outputs]
        out_alt  = weights[table_ix, lookup_alt_indices]
        grad_view = grad_out.unsqueeze(2)                 # [B, n_heads, 1, n_outputs]
        grad_main = (out_main.view(B, n_heads, tph, n_outputs) * grad_view).sum(-1) \
                    .view(B, n_lookup_tables).contiguous()
        grad_alt  = (out_alt.view(B, n_heads, tph, n_outputs)  * grad_view).sum(-1) \
                    .view(B, n_lookup_tables).contiguous()

        flat_lookup = lookup_indices.reshape(-1)
        table_offset = (
            torch.arange(n_lookup_tables, device=weights.device, dtype=lookup_indices.dtype) * table_dim
        ).unsqueeze(0).expand(B, -1).reshape(-1)
        fully_flat_idx = table_offset + flat_lookup
        grad_per_lookup = (
            grad_out.unsqueeze(2)
                    .expand(B, n_heads, tph, n_outputs)
                    .reshape(B * n_lookup_tables, n_outputs)
                    .contiguous()
        )
        grad_weights_flat = torch.zeros(
            n_lookup_tables * table_dim, n_outputs,
            dtype=weights.dtype, device=weights.device,
        )
        grad_weights_flat.index_add_(0, fully_flat_idx, grad_per_lookup)
        grad_weights = grad_weights_flat.view(n_lookup_tables, table_dim, n_outputs)

        return grad_weights, None, None, grad_main, grad_alt, None, None


class TinyMultiHeadLut(nn.Module):
    """Multi-head LUT with TinyAnchorPairsLookup + bf16 (default) weights.

    Args:
        input_dim: Dimension of input tensor (must be <= 32767).
        n_heads: Number of heads.
        n_outputs: Number of output dimensions per head.
        n_anchor_pairs: Number of anchor pairs per table (1..15).
        tables_per_head: Number of lookup tables per head.
        weight_dtype: Storage dtype for weights (default torch.bfloat16).
        anchor_sampling_policy: CANONICAL_FULL_COVERAGE (default) or
            CANONICAL_DISTINCT.
        partition_sets: Optional list-of-lists restricting CANONICAL_DISTINCT
            sampling to within-partition pairs.
        random_seed: Seed for anchor sampling and weight init.
        initial_weights_noise: Uniform [-σ, +σ] init for weights (default 0.001).
        device: torch.device or None.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        *,
        weight_dtype: torch.dtype = torch.bfloat16,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        partition_sets: Optional[list] = None,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not (1 <= n_anchor_pairs <= 15):
            raise ValueError(
                f"TinyMultiHeadLut requires 1 <= n_anchor_pairs <= 15 "
                f"(int16 lookup-index range), got {n_anchor_pairs}"
            )
        if input_dim > 32767:
            raise ValueError(
                f"TinyMultiHeadLut requires input_dim <= 32767 (int16 anchor "
                f"index range), got {input_dim}"
            )

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.table_dim = 1 << n_anchor_pairs  # 2 ** n_anchor_pairs
        self.weight_dtype = weight_dtype

        n_lookup_tables = n_heads * tables_per_head
        self.n_lookup_tables = n_lookup_tables

        # Anchor lookup (int16 path).
        self.lookup = TinyAnchorPairsLookup(
            input_dim=input_dim,
            n_tables=n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            n_heads=n_heads,
            random_seed=random_seed,
            device=device,
            partition_sets=partition_sets,
            anchor_sampling_policy=anchor_sampling_policy,
        )

        # Weights: [n_lookup_tables, table_dim, n_outputs] in weight_dtype.
        # Init: uniform[-σ, +σ] cast to weight_dtype.
        dev = device or torch.device("cpu")
        rng_kwargs: dict = {"device": dev}
        if random_seed is not None:
            rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed + 1)
        weights_init = (
            (torch.rand(n_lookup_tables, self.table_dim, n_outputs, **rng_kwargs) - 0.5)
            * (2.0 * initial_weights_noise)
        ).to(weight_dtype)
        self.weights = nn.Parameter(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float [B, input_dim]
        Returns:
            [B, n_heads, n_outputs] in weight_dtype.
        """
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )

        # TinyAnchorPairsLookup returns BOTH the chosen ("main") and runner-up
        # ("alt") int16 lookup indices, plus their zero-valued carriers
        # (lookup_indices_grad_c, lookup_alt_indices_grad_c) whose gradients
        # back-flow through the anchor STE kernel into x.grad. We must thread
        # both carriers through our autograd Function — dropping the alt one
        # silently breaks numerical equivalence with MultiHeadLut.
        (lookup_indices, lookup_alt_indices, _alt_deltas,
         lookup_indices_grad_c, lookup_alt_indices_grad_c) = self.lookup(x)
        # TAPL returns lookup_alt_indices with a trailing n_alt=1 dim
        # (multi-alt API parity); squeeze for our na=1 path.
        lookup_indices = lookup_indices.to(torch.int64)
        lookup_alt_indices = lookup_alt_indices.squeeze(-1).to(torch.int64)

        # Eval / no_grad path: TAPL returns None carriers; nothing to
        # backprop, so skip the autograd Function and just do the
        # embedding_bag forward directly.
        if lookup_indices_grad_c is None:
            return _embedding_bag_forward(
                self.weights, lookup_indices, self.n_heads, self.tables_per_head,
            )

        # Training path: thread BOTH carriers through the autograd Function
        # so its backward returns grad_main AND grad_alt. Dropping the alt
        # carrier silently breaks numerical equivalence with MultiHeadLut.
        return _TinyMHLutGatherReduce.apply(
            self.weights, lookup_indices, lookup_alt_indices,
            lookup_indices_grad_c, lookup_alt_indices_grad_c.squeeze(-1),
            self.n_heads, self.tables_per_head,
        )
