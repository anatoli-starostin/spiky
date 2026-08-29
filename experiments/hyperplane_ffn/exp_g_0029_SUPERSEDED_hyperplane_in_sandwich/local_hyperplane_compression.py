"""HyperplaneCompressionMHL — CompressionMultiHeadLUT with HyperplaneMHL inside.

LOCAL to exp_g_0029. Shared `src/spiky/lutorch/` is NOT touched.

Why a wrapper is needed at all
------------------------------
`HyperplaneMultiHeadLUT` is a generalization of `FastMultiHeadLut`, not of
`CompressionMultiHeadLUT`. CompressionMHL is a sandwich —

    compress Linear(input_dim -> n_heads*inner_in)
      -> per-head LUT over each head's own inner_in slice
      -> decompress Linear(n_heads*inner_out -> output_dim)

— and only the middle layer is a FastMHL. So "swap CompressionMHL for
HyperplaneMHL" means: keep the sandwich exactly, replace the filling. That is what
this class does, and it is the only way to hold every other hyperparameter
(n_heads, inner_in, inner_out, nap, tph) identical to exp_n_0121.

Which CompressionMHL path this mirrors
--------------------------------------
CompressionMHL has two numerically-equivalent implementations of its INDEPENDENT
per-head path: a `nn.ModuleList` loop of single-head FastMHLs, and a single batched
FastMHL with `multi_head_input=True` and block-diagonal anchors. exp_n_0121 ran the
batched one (it is the default whenever `has_compress`).

`HyperplaneMultiHeadLUT` has no `multi_head_input` mode — its forward takes
`x: [B, input_dim]` and gives every head the SAME input. Feeding it the shared
compressed vector would make each head read all 4*48 dims, which is
`joint_head_compression=True`, an architecture exp_n_0121 does not have. So this
wrapper mirrors the LOOP path instead: one single-head HyperplaneMHL per head, each
reading its own inner_in slice, seeded `random_seed + h` — the exact convention
CompressionMHL's ModuleList uses.

The one consequence, stated honestly: the loop gives each head its own
(log_soft_score_temp, log_select_temp) — 8 temps per slot at H=4 — where the batched
path shares one pair, 2 per slot. That is +6 params per slot, +36 over 6 slots. It is
unavoidable without a batched HyperplaneMHL, and it is 36 parameters out of ~68.5M.
(It is also the exact discrepancy visible between exp_n_0121's summary.json and its
flops_bandwidth.txt, for the same reason.)

Init
----
`hyperplane_init="anchor_pairs"` (the default here) initializes each hyperplane to
reproduce its fixed anchor pair EXACTLY — `w_i = e_a - e_b`, `b_i = 0`, no noise — so
at step 0 the module is bit-for-bit the FastMHL that exp_n_0121 started from. The two
runs therefore begin from the same selection function and diverge only as the
hyperplanes learn, which is what makes the step-for-step A/B meaningful.
"""
import torch
import torch.nn as nn

from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT


class HyperplaneCompressionMHL(nn.Module):
    """Drop-in for CompressionMultiHeadLUT with HyperplaneMHL as the inner LUT.

    Accepts CompressionMultiHeadLUT's keyword signature so the call site in
    train.py changes by class name only. Options this wrapper cannot honour
    faithfully raise rather than silently changing the architecture.

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
        random_seed=None,
        device=None,
    ):
        super().__init__()
        # Refuse the configurations this wrapper would have to fake. exp_n_0121 uses
        # none of them, so these are guards, not limitations in practice.
        if joint_head_compression:
            raise NotImplementedError(
                "joint_head_compression is not supported: HyperplaneMHL has no "
                "multi_head_input mode, so the joint path cannot be mirrored faithfully."
            )
        if inner_residual or pre_lut_meanabsnorm:
            raise NotImplementedError(
                "inner_residual / pre_lut_meanabsnorm are not mirrored by this wrapper."
            )
        in_raw = inner_in_dim if inner_in_dim is not None else inner_dim
        out_raw = inner_out_dim if inner_out_dim is not None else inner_dim
        if in_raw is None or out_raw is None:
            raise ValueError("inner dims must be given (inner_dim or inner_in/out_dim)")
        if in_raw == -1:
            raise NotImplementedError(
                "inner_in_dim=-1 (no compress) is not supported: without per-head "
                "compression every head reads the same input, which is the joint path."
            )

        eff_in = in_raw
        eff_out = output_dim if out_raw == -1 else out_raw

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_in_dim = in_raw
        self.inner_out_dim = out_raw
        self.eff_in, self.eff_out = eff_in, eff_out
        self.n_heads = n_heads
        self.nap, self.tph = nap, tph
        self.has_compress = True
        self.has_decompress = out_raw != -1
        self.joint_head_compression = False
        # Mirrors CompressionMHL's ModuleList path; there is no batched HyperplaneMHL.
        self.batched_multi_head_input = False
        self.hyperplane_init = hyperplane_init

        self.compress = nn.Linear(input_dim, n_heads * in_raw, device=device)
        self.luts = nn.ModuleList([
            HyperplaneMultiHeadLUT(
                input_dim=eff_in, n_heads=1, n_outputs=eff_out,
                n_anchor_pairs=nap, tables_per_head=tph,
                forward_mode=forward_mode, weight_dtype=weight_dtype,
                use_bf16=use_bf16, initial_weights_noise=initial_weights_noise,
                learnable_temps=learnable_temps,
                hyperplane_init=hyperplane_init,
                hyperplane_init_scale=hyperplane_init_scale,
                # same seed convention as CompressionMHL's per-head ModuleList
                random_seed=(None if random_seed is None else random_seed + h),
                device=device,
            )
            for h in range(n_heads)
        ])
        self.decompress = (nn.Linear(n_heads * out_raw, output_dim, device=device)
                           if self.has_decompress else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [N, {self.input_dim}], got {tuple(x.shape)}"
            )
        N = x.shape[0]
        z = self.compress(x).view(N, self.n_heads, self.inner_in_dim)   # [N, H, inner_in]
        parts = []
        for h, lut in enumerate(self.luts):
            z_h = z[:, h, :]                                  # [N, eff_in]
            parts.append(lut(z_h).sum(dim=1).to(z_h.dtype))   # [N, 1, eff_out] -> [N, eff_out]
        if self.has_decompress:
            return self.decompress(torch.cat(parts, dim=-1))  # [N, H*inner_out] -> [N, out]
        return sum(parts)
