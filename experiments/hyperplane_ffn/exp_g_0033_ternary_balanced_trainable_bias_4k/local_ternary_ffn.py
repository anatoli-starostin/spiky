"""PureTernaryHyperplaneMHL — TernaryHyperplaneMHL as a DIRECT FFN LUT slot.

LOCAL to exp_g_0030. Identical to exp_g_0029's local_hyperplane_ffn.py in every
respect EXCEPT the class it instantiates: TernaryHyperplaneMultiHeadLUT instead of
HyperplaneMultiHeadLUT. Same no-sandwich topology, same full-model-dim cells, same
head-sum plumbing. The ternary class is a subclass of HyperplaneMultiHeadLUT living in
the shared library (added, not modified) -- so the routing math, the soft backward and
the table lookup are the same code; only the hyperplane weights reaching the
projection are quantized to {-1, 0, +1} through a straight-through estimator.

Two differences follow automatically from the ternary class and are NOT choices made
here: it has no bias (the routing test is <q, x> > 0, and hyperplane_bias is a frozen
zero buffer rather than a Parameter), and it adds one trainable log_ternary_temp per
table.

No compression. No decompression. The LUT reads the full model dimension and every
table cell stores a full model-dim output vector:

    x [N, 384]  ->  HyperplaneMultiHeadLUT(input_dim=384, n_outputs=384)  ->  [N, H, 384]
                ->  sum over heads                                        ->  [N, 384]

Contrast with exp_n_0121, which wrapped its FastMHL in a sandwich —
`compress Linear(384 -> H*48)` → per-head LUT over 48 dims → `decompress Linear(H*48 -> 384)`
— so its cells stored 48-dim vectors. Removing both projections is what makes the
tables grow ~8x (48-dim cells -> 384-dim cells): the table is
`n_heads * tph * 2^nap * n_outputs`, and only `n_outputs` changes, 48 -> 384.

Each of the `n_heads * tph` tables picks its row with `nap` learned affine sign tests
`1[<w_i, x> + b_i > 0]` evaluated on the FULL 384-dim input, so the hyperplanes are
[n_tables, nap, 384] rather than [n_tables, nap, 48].

Two consequences worth stating rather than discovering later:

1. **No zero-init handle.** exp_n_0121's slot output was exactly zero at step 0
   because train.py zero-initializes `decompress.weight`. With no decompress there is
   nothing to zero, so the slot starts at the LUT tables' own init noise: a sum of
   `n_heads * tph = 512` cells each Uniform[-1e-3, 1e-3], i.e. std ~0.013 per output
   dim. Small against LayerNorm'd activations but not identically zero. This is
   inherent to removing the projection, not a choice. `has_decompress = False` is
   exposed so train.py's zero-init block skips it cleanly.

2. **Heads share the input.** With no per-head compress every head reads the same x,
   which is the `joint_head_compression=True` topology, and the head outputs are
   summed. That is forced: without a compress there are no per-head slices to give
   them, and HyperplaneMHL has no `multi_head_input` mode.
"""
import torch
import torch.nn as nn

from spiky.lutorch.ternary_hyperplane_multi_head_lut import (
    TernaryHyperplaneMultiHeadLUT,
)


class PureTernaryHyperplaneMHL(nn.Module):
    """TernaryHyperplaneMHL at full model dim, no compress/decompress.

    Accepts CompressionMultiHeadLUT's keyword signature so the call site in train.py
    changes by class name only. `inner_in_dim` / `inner_out_dim` must be -1 (the
    codebase's existing "no projection" sentinel); any other value would mean a
    sandwich, which is exactly what this module exists to remove, so it raises rather
    than silently ignoring it.

    Forward: x [N, input_dim] -> [N, output_dim].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inner_dim=None,
        *,
        inner_in_dim=None,
        inner_out_dim=None,
        nap: int,
        tph: int,
        n_heads: int = 1,
        inner_residual: bool = False,
        joint_head_compression: bool = False,
        forward_mode: str = "hard",
        weight_dtype: torch.dtype = torch.float32,
        use_bf16: bool = False,
        initial_weights_noise: float = 1e-3,
        learnable_temps: bool = True,
        pre_lut_meanabsnorm: bool = False,
        batched_multi_head_input: bool = True,
        hyperplane_init: str = "anchor_pairs",
        hyperplane_init_scale=None,
        ternary_temp_init: float = 0.5,
        trainable_bias: bool = False,
        random_seed=None,
        device=None,
    ):
        super().__init__()
        in_raw = inner_in_dim if inner_in_dim is not None else inner_dim
        out_raw = inner_out_dim if inner_out_dim is not None else inner_dim
        if in_raw != -1 or out_raw != -1:
            raise ValueError(
                "PureHyperplaneMHL is the no-sandwich variant: it requires "
                f"inner_in_dim == inner_out_dim == -1, got in={in_raw}, out={out_raw}. "
                "A non -1 inner dim would reintroduce the compress/decompress Linears."
            )
        if inner_residual or pre_lut_meanabsnorm:
            raise NotImplementedError(
                "inner_residual / pre_lut_meanabsnorm are not mirrored by this module."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_in_dim = self.inner_out_dim = -1
        self.eff_in, self.eff_out = input_dim, output_dim
        self.n_heads = n_heads
        self.nap, self.tph = nap, tph
        # No projections at all -- train.py reads these to decide what to zero-init.
        self.has_compress = False
        self.has_decompress = False
        self.compress = nn.Identity()
        self.decompress = nn.Identity()
        # Every head reads the full x; the head outputs are summed.
        self.joint_head_compression = True
        self.batched_multi_head_input = False
        self.hyperplane_init = hyperplane_init

        self.ternary_temp_init = ternary_temp_init
        self.trainable_bias = bool(trainable_bias)
        self.lut = TernaryHyperplaneMultiHeadLUT(
            input_dim=input_dim, n_heads=n_heads, n_outputs=output_dim,
            n_anchor_pairs=nap, tables_per_head=tph,
            forward_mode=forward_mode, weight_dtype=weight_dtype,
            use_bf16=use_bf16, initial_weights_noise=initial_weights_noise,
            learnable_temps=learnable_temps,
            hyperplane_init=hyperplane_init,
            hyperplane_init_scale=hyperplane_init_scale,
            ternary_temp_init=ternary_temp_init,
            trainable_bias=trainable_bias,
            random_seed=random_seed, device=device,
        )
        # The class warns if an init quantizes to an all-zero routing; make it fatal
        # here rather than training a dead model for hours.
        nz = int((self.lut.hard_ternary_weight() != 0).sum())
        if nz == 0:
            raise RuntimeError(
                f"ternary routing is ENTIRELY ZERO at init (T={ternary_temp_init}, "
                f"dead zone |w| <= T*ln3). Refusing to build."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [N, {self.input_dim}], got {tuple(x.shape)}"
            )
        return self.lut(x).sum(dim=1).to(x.dtype)      # [N, H, out] -> [N, out]
