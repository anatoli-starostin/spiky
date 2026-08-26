"""LOCAL override: use the BATCHED multi_head_input FastMHL even with no compress.

RETAINED BUT **NOT USED** -- documented negative work. train.py imports the STOCK
CompressionMultiHeadLUT instead, for two measured reasons:

  1. WRONG PARAMETERIZATION for this ablation. Batching collapses the loop's
     per-head (log_soft_score_temp, log_select_temp) -- 8 pairs, 16 scalars per
     slot -- into ONE shared pair (2 per slot), -84 params over 6 layers. The
     exp_n_0045 checkpoint this experiment ablates has the PER-HEAD temps
     (93,403,488 params; its profiler output says "path=per-head-loop", and the
     run predates the batched path by one day: 2026-08-19 vs 3328bf5c 2026-08-20).
     Using this class would change the temperature parameterization on top of the
     compress ablation, making it a two-variable experiment.
  2. IT COSTS MEMORY, it does not save it. The backward surrogate becomes ONE
     [N, 2048, 128] tensor instead of 8 sequential [N, 256, 128] ones: 6.44 GB at
     device_bs 12 and 25.77 GB at 48, versus the loop's ~4.25 GB total peak at 12.
     So batching would also have forced a batch-size difference from exp_n_0045.

The code below is correct and verified (max |loop - batched| = 0.000e+00 over a
64-row batch, table init and block-diagonal anchors matching head-for-head); it is
kept because the "batch the heads" idea is a reasonable thing to reach for again,
and this records why it is the wrong tool for THIS experiment.

The shared CompressionMultiHeadLUT gates its batched path on `has_compress`:

    self.batched_multi_head_input = (requested and not joint and self.has_compress)

The reason is what the batched FastMHL consumes. With `multi_head_input=True` it
expects a per-head-stacked input -- [N, n_heads, input_dim], flattened to
[N, n_heads*input_dim] -- and its anchors are BLOCK-DIAGONAL: head h reads only
columns [h*input_dim, (h+1)*input_dim). In the normal path the compress Linear
produces exactly that stack (each head its own inner_in slice), so there is
something to feed it. With inner_in_dim=-1 there is no compress, every head reads
the SAME full input_dim vector, and the class falls back to a python loop over
n_heads separate single-head FastMHLs.

That fallback is correct but wasteful here: 8 sequential kernel launches per slot
per forward AND per backward, where one batched call would do.

THE FIX, and why it is numerically equivalent rather than merely similar:
feeding `x.expand(N, n_heads, input_dim)` gives head h a block that is a copy of
the same x -- which is precisely what the loop hands each of its 8 LUTs. The
block-diagonal anchors then index within each head's own copy, so head h sees the
identical 384-d vector it saw in the loop. The shared class already builds its
batched module with `random_seed=random_seed` and documents that it reproduces
the loop's `seed + h` convention internally, so the anchors and table init match
head-for-head. Equivalence is asserted at runtime in verify_batched.py rather
than taken on trust.

ONE REAL DIFFERENCE, NOT AN EXECUTION DETAIL: the loop builds n_heads separate
FastMHL modules, each owning its own (log_soft_score_temp, log_select_temp) pair.
The batched module is ONE FastMHL and owns ONE pair. So batching collapses
per-head learnable temperatures into per-layer shared ones: 8*2=16 -> 2 per slot,
i.e. -14 per slot and -84 across 6 layers. Parameter count therefore does NOT
stay identical -- it drops by 84. This is the same effect seen when nebius made
the batched path the CompressionMHL default (exp_n_0033 27,343,296 vs exp_g_0006
27,343,200, also exactly the temps).

Shared src/spiky/lutorch/ is NOT modified; exp_n_0045 keeps importing the stock
class and is unaffected.
"""
import torch
import torch.nn as nn

from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut


class BatchedNoCompressMHL(CompressionMultiHeadLUT):
    """CompressionMultiHeadLUT that batches the heads even when has_compress is False."""

    def __init__(self, *args, **kwargs):
        # Build the stock module first (it lands on the per-head loop path), then
        # replace that loop with a single batched FastMHL. Reusing the parent's
        # __init__ keeps every other detail -- decompress, dtypes, seeds, temps
        # policy -- exactly as the shared class defines it.
        kwargs = dict(kwargs)
        kwargs['batched_multi_head_input'] = False
        random_seed = kwargs.get('random_seed')
        super().__init__(*args, **kwargs)

        self._local_batched_nocompress = False
        if self.joint_head_compression or self.has_compress or self.n_heads == 1:
            # Nothing to gain: either the stock class already batches, or there is
            # only one head. Leave it exactly as the parent built it.
            return

        proto = self.luts[0]
        self.lut_batched = FastMultiHeadLut(
            input_dim=self.eff_in,
            n_heads=self.n_heads,
            n_outputs=self.eff_out,
            n_anchor_pairs=proto.n_anchor_pairs,
            tables_per_head=proto.tables_per_head,
            multi_head_input=True,
            forward_mode=proto.forward_mode,
            weight_dtype=proto.weight_dtype,
            use_bf16=proto.use_bf16,
            learnable_temps=proto.learnable_temps,
            random_seed=random_seed,
            device=proto.weights.device,
        )
        del self.luts                      # drop the loop entirely -- no dead params
        self._local_batched_nocompress = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._local_batched_nocompress:
            return super().forward(x)

        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x shape must be [N, {self.input_dim}], got {tuple(x.shape)}")
        N = x.shape[0]
        # Every head routes on the SAME full input -- exactly what the per-head loop
        # did. expand() is a view; the batched FastMHL reshapes it to the flat
        # [N, n_heads*input_dim] block layout its anchors expect.
        z3 = x.unsqueeze(1).expand(N, self.n_heads, self.input_dim)
        if self.pre_lut_meanabsnorm:
            z3 = z3 / (z3.abs().mean(-1, keepdim=True) + 1e-6)
        y = self.lut_batched(z3).to(x.dtype)          # [N, n_heads, eff_out]
        if self.inner_residual:
            y = y + z3
        if self.has_decompress:
            return self.decompress(y.reshape(N, self.n_heads * self.eff_out))
        return y.sum(dim=1)
