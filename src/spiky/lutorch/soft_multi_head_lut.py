"""SoftMultiHeadLUT — fully differentiable, matmul-only MultiHeadLut.

Replaces the discrete pair-comparison + table-row gather of MultiHeadLut with a
soft attention over all 2^nap rows, computed entirely from matmul + softmax +
elementwise ops. Friendly to fp8 / Tensor Core kernels and smooth in both the
pair signs and the row selector. As the two temperatures (T_soft, T_sel) tend
to zero, the output converges to the discrete MultiHeadLut lookup.

Pipeline (per token, per table, see `forward` for shapes):

    1.  rd  = x[a] - x[b]                                              # raw pair distances
    2.  p   = rd / (T_soft + |rd|)                                     # soft signs in (-1, +1)
    3.  ts  = p @ bit_matrix                                           # row match scores (popcount-equivalent at hard limit)
    4.  sel = softmax(ts / T_sel)        (or gumbel_softmax)            # soft row selector
    5.  o_t = sel @ weights[t]                                         # per-table soft lookup
    6.  o   = sum_{t in head} o_t                                      # head sum

Anneal `T_sel` toward zero on a schedule to harden the lookup; keep `T_soft`
small (~0.01) so the pair signs are already near-binary at init.

Cost: materialises [B, n_tables, 2^nap] in step 3-4. Practical for nap ≤ 8.
"""
from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs


def _bit_matrix(dim: int, device: Optional[torch.device] = None,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[dim, 2^dim] - column k is the +/-1 binary expansion of integer k (MSB top)."""
    n = 1 << dim
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(dim - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


class SoftMultiHeadLUT(nn.Module):
    """Soft, matmul-only MultiHeadLut. See module docstring for the pipeline.

    Args:
        input_dim: Number of input features.
        n_outputs: Output dim per head.
        n_anchor_pairs: Pairs monitored per table (nap). Sets table_dim = 2^nap.
        n_heads: Number of heads.
        tables_per_head: Tables per head; total tables = n_heads * tables_per_head.
        soft_score_temp: T_soft - rational soft-sign temperature. Default 0.01,
            i.e. signs are near-±1 for any non-trivial pair distance.
        select_temp: T_sel - row-softmax temperature. Buffer, can be annealed.
        gumbel: If True, use gumbel_softmax for the row selector.
        hard: If True, use straight-through estimator — forward is exactly
            one-hot at argmax(sel), backward uses the soft sel for gradients.
            Combines with gumbel: gumbel=True+hard=True uses
            F.gumbel_softmax(hard=True); gumbel=False+hard=True uses
            argmax of the temperature-scaled softmax with manual STE.
        learnable_temps: If True, T_soft and T_sel become trainable
            Parameters parametrized as exp(log_T) (so they stay positive).
            Initialized at the supplied soft_score_temp / select_temp values.
            One scalar Parameter per temperature per LUT instance.
        anchor_sampling_policy: Anchor pair sampling policy (default
            CANONICAL_FULL_COVERAGE), forwarded to get_balanced_anchor_pairs.
        partition_sets: Optional list of index sets restricting pair sampling
            to within-partition pairs (forbids cross-partition pairs).
        initial_weights_noise: Uniform[-sigma, sigma] init for table weights.
        weight_dtype: Dtype for the LUT weights (Parameter). Default fp32.
        random_seed: For anchor pair sampling and weight init.
        device: Target device.
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        n_anchor_pairs: int,
        n_heads: int,
        tables_per_head: int,
        soft_score_temp: float = 0.01,
        select_temp: float = 1.0,
        gumbel: bool = False,
        hard: bool = False,
        learnable_temps: bool = False,
        use_bf16: bool = False,
        compile_forward: bool = False,
        anchor_sampling_policy: AnchorSamplingPolicy = AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        partition_sets: Optional[list] = None,
        initial_weights_noise: float = 0.001,
        weight_dtype: torch.dtype = torch.float32,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if n_anchor_pairs < 1:
            raise ValueError(f"n_anchor_pairs must be >= 1, got {n_anchor_pairs}")
        if n_anchor_pairs > 12:
            # 2^12 = 4096 rows per table; beyond this memory blows up fast.
            raise ValueError(
                f"n_anchor_pairs={n_anchor_pairs} > 12; soft path materialises "
                f"[B, n_tables, 2^nap]. Use the discrete MultiHeadLut for larger nap."
            )

        self.input_dim = input_dim
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.n_heads = n_heads
        self.tables_per_head = tables_per_head
        self.n_lookup_tables = n_heads * tables_per_head
        self.table_dim = 1 << n_anchor_pairs
        self.gumbel = bool(gumbel)
        self.hard = bool(hard)
        self.learnable_temps = bool(learnable_temps)
        self.use_bf16 = bool(use_bf16)
        self.weight_dtype = weight_dtype

        if self.learnable_temps:
            # Log-parametrize so the optimizer sees an unconstrained scalar
            # while the effective temperature stays positive. One scalar
            # Parameter per temperature per LUT instance.
            self.log_soft_score_temp = nn.Parameter(
                torch.tensor(math.log(float(soft_score_temp)), dtype=torch.float32)
            )
            self.log_select_temp = nn.Parameter(
                torch.tensor(math.log(float(select_temp)), dtype=torch.float32)
            )
            # Keep static initial values for printing only.
            self._init_soft_score_temp = float(soft_score_temp)
            self._init_select_temp = float(select_temp)
        else:
            # Fixed temperatures: T_soft as Python float, T_sel as 0-d buffer
            # so .set_select_temp() can mutate it without re-instantiation.
            self.soft_score_temp = float(soft_score_temp)
            self.register_buffer(
                "select_temp", torch.tensor(float(select_temp), dtype=torch.float32)
            )

        dev = device or torch.device("cpu")

        # Anchor pairs via project sampler (matches MHLut/TinyMHLut semantics).
        anchor_a, anchor_b = get_balanced_anchor_pairs(
            n_tables=self.n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            input_dim=input_dim,
            device=dev,
            random_seed=random_seed,
            policy=anchor_sampling_policy,
            n_heads=n_heads,
            shuffle_per_head=True,
            partition_sets=partition_sets,
        )
        self.register_buffer("anchor_pairs_a", anchor_a.long())  # [n_tables, nap]
        self.register_buffer("anchor_pairs_b", anchor_b.long())

        # Precomputed +/-1 bit matrix: [nap, 2^nap].
        self.register_buffer("bit_matrix", _bit_matrix(n_anchor_pairs, device=dev))

        # LUT weights: [n_tables, table_dim, n_outputs] uniform[-sigma, sigma].
        rng_kwargs: dict = {"device": dev}
        if random_seed is not None:
            rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed + 1)
        weights_init = (
            (torch.rand(self.n_lookup_tables, self.table_dim, n_outputs, **rng_kwargs) - 0.5)
            * (2.0 * initial_weights_noise)
        ).to(weight_dtype)
        self.weights = nn.Parameter(weights_init)

        # Optional torch.compile of the forward pass — fuses einsum -> softmax ->
        # einsum into a single graph, materially faster than eager. dynamic=True
        # so we don't recompile when batch size changes (train vs eval).
        if compile_forward:
            self.forward = torch.compile(self.forward, dynamic=True)

    def set_select_temp(self, new_T_sel: float) -> None:
        """External-schedule entry point. Mutates the buffer/Parameter in place."""
        if self.learnable_temps:
            with torch.no_grad():
                self.log_select_temp.fill_(math.log(float(new_T_sel)))
        else:
            self.select_temp.fill_(float(new_T_sel))

    def _temps(self):
        """Return the effective (T_soft, T_sel). Either Python floats / buffer
        tensor (fixed mode) or live differentiable tensors (learnable mode)."""
        if self.learnable_temps:
            return self.log_soft_score_temp.exp(), self.log_select_temp.exp()
        return self.soft_score_temp, self.select_temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        returns: [B, n_heads, n_outputs] in weight_dtype
        """
        B = x.shape[0]
        # Optional bf16 autocast for the soft pipeline. Inside autocast, the
        # two einsums route to bf16 Tensor Core matmuls (~2x speedup, halved
        # activation memory); softmax stays in fp32 by autocast policy.
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if (self.use_bf16 and x.is_cuda)
            else contextlib.nullcontext()
        )
        T_soft, T_sel = self._temps()
        with autocast_ctx:
            # 1) per-table pair distances
            x_a = x[:, self.anchor_pairs_a]                              # [B, n_tables, nap]
            x_b = x[:, self.anchor_pairs_b]
            rd = x_a - x_b

            # 2) soft signs in (-1, +1)
            p = rd / (T_soft + rd.abs())                                 # [B, n_tables, nap]

            # 3) row match scores against the +/-1 bit pattern of each row index.
            #    ts[b, t, k] = sum_i p[b, t, i] * bit_matrix[i, k]
            ts = torch.einsum("btp,pk->btk", p, self.bit_matrix.to(p.dtype))  # [B, n_tables, 2^nap]

            # 4) row selector
            if self.gumbel:
                # F.gumbel_softmax requires scalar tau; this triggers a
                # graph break under torch.compile but works correctly.
                tau = float(T_sel.detach()) if torch.is_tensor(T_sel) else float(T_sel)
                sel = F.gumbel_softmax(ts, tau=tau, hard=self.hard)
            else:
                sel_soft = F.softmax(ts / T_sel, dim=-1)
                if self.hard:
                    # Straight-through: forward is one-hot at argmax,
                    # backward sees the gradient of sel_soft.
                    index = sel_soft.argmax(dim=-1, keepdim=True)
                    sel_hard = torch.zeros_like(sel_soft).scatter_(-1, index, 1.0)
                    sel = sel_hard - sel_soft.detach() + sel_soft
                else:
                    sel = sel_soft

            # 5) per-table soft lookup
            out_t = torch.einsum("btk,tko->bto", sel, self.weights)      # [B, n_tables, n_outputs]

        # 6) sum over the tph tables of each head, in weight dtype
        out_t = out_t.to(self.weights.dtype)
        return out_t.view(B, self.n_heads, self.tables_per_head, self.n_outputs).sum(dim=2)
