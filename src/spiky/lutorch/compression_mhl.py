"""CompressionMultiHeadLUT (short: CompressionMHL) — a compress / LUT / decompress bottleneck.

Wraps a FastMultiHeadLut inside a linear compress→decompress pair so the (expensive,
sparse-gradient) table lookup can operate in a chosen inner space rather than the full
input/output space:

    z   = compress(x)          # input_dim  -> inner_in_dim    (dense Linear)
    y   = lut(z)               # FastMHL     inner_in_dim -> inner_out_dim
    out = decompress(y)        # inner_out_dim -> output_dim    (dense Linear)

The LUT's input width (`inner_in_dim`) and output width (`inner_out_dim`) are set
independently. Either projection can be dropped with a `-1` sentinel:
  * inner_in_dim = -1  -> NO compress; the LUT reads x directly (its input_dim = input_dim).
  * inner_out_dim = -1 -> NO decompress; the LUT emits output_dim and heads are summed to
    the final output.
  * both -1            -> a pure FastMHL FFN slot (no projections).
The legacy `inner_dim` kwarg is still accepted and sets both to the same value.

Reused across the CompressionMHL experiment series (which varies the inner dims and `tph`).
See `CompressionMultiHeadLUT.param_count(...)` for the exact parameter formula.

INIT-SEEDING CAVEAT for `inner_in_dim=-1` + `joint_head_compression=False`
-------------------------------------------------------------------------
The independent path is served by a single batched FastMHL. When there is no compress, all
heads share the same input, so that module is an ordinary shared-input multi-head FastMHL
seeded once from `random_seed`. An earlier revision ran this case through a per-head loop of
single-head LUTs seeded `random_seed + h`, so **models built in that configuration no longer
initialise identically to ones built before**. Shapes, forward/backward semantics and
parameter counts are unchanged -- only the initial anchor/table draw differs.

This affects exactly one published configuration: `exp_n_0138` (paper section 5 / Table 3,
output-compression-only, reported 1.21249 bpb), whose number came from the old seeding and
is therefore not bit-reproducible from this code. Every other configuration in the paper
sets `inner_in_dim` to a real width and is unaffected (the block-diagonal `random_seed + h`
convention is preserved there).
"""
from typing import Optional

import torch
import torch.nn as nn

from spiky.lutorch.bh4_multi_head_lut import BH4MultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT


def _resolve_inner(inner_dim, inner_in_dim, inner_out_dim):
    """Resolve the (inner_in, inner_out) pair from the legacy `inner_dim` shim or the two
    explicit dims. Returns the RAW values (which may be the -1 'no projection' sentinel)."""
    if inner_dim is not None:
        if inner_in_dim is not None or inner_out_dim is not None:
            raise ValueError(
                "pass either inner_dim (sets both) OR inner_in_dim/inner_out_dim, not both"
            )
        return inner_dim, inner_dim
    if inner_in_dim is None or inner_out_dim is None:
        raise ValueError("pass inner_dim, or BOTH inner_in_dim and inner_out_dim")
    return inner_in_dim, inner_out_dim


class CompressionMultiHeadLUT(nn.Module):
    """Linear-compress → FastMultiHeadLut → linear-decompress bottleneck.

    Args:
        input_dim: dimension of x.
        output_dim: dimension of the returned vector.
        inner_dim: legacy convenience — sets inner_in_dim = inner_out_dim = inner_dim.
            Mutually exclusive with inner_in_dim/inner_out_dim.
        inner_in_dim: the LUT's input width. -1 means "no compress" (LUT reads x directly,
            so its input width = input_dim).
        inner_out_dim: the LUT's per-head output width. -1 means "no decompress" (LUT emits
            output_dim and heads are summed to the output directly).
        nap: FastMHL n_anchor_pairs (K = 2**nap rows per table).
        tph: FastMHL tables_per_head.
        n_heads: FastMHL output heads (default 1). Heads are summed before decompress.
        inner_residual: if True, add the LUT input to its output (`y = lut(z) + z`). Only
            valid when the LUT's effective input width == effective output width (the skip
            lives in one space); raises otherwise. Adds ZERO parameters.
        joint_head_compression: at n_heads>1, whether compress/decompress are shared across
            heads. True -> JOINT (one shared pair + one FastMHL(n_heads)); False (DEFAULT) ->
            INDEPENDENT (per-head compress/decompress + one batched FastMHL whose heads
            route block-diagonally over their own inner_in slices).
            At n_heads=1 the two modes are numerically identical.
        forward_mode: "hard" (default) or "hybrid_smooth"; passed to FastMHL.
        backward_topk: 0 (default, full-K soft surrogate) or >0 for the
            sparse-Hamming ("soft_topk") backward; passed to FastMHL (fast
            lut_impl only; ignored on the light path).
        weight_dtype: FastMHL table storage dtype (default fp32).
        use_bf16: FastMHL bf16-autocast flag (default False — these experiments run fp32).
        initial_weights_noise: FastMHL near-zero table init (default 1e-3).
        random_seed: FastMHL anchor/table seed.
        device: optional device for the submodules.

    Forward:
        x: float [N, input_dim]  ->  [N, output_dim].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inner_dim: Optional[int] = None,
        *,
        inner_in_dim: Optional[int] = None,
        inner_out_dim: Optional[int] = None,
        nap: int,
        tph: int,
        n_heads: int = 1,
        inner_residual: bool = False,
        joint_head_compression: bool = False,
        forward_mode: str = "hard",
        backward_topk: int = 0,
        weight_dtype: torch.dtype = torch.float32,
        use_bf16: bool = False,
        initial_weights_noise: float = 1e-3,
        learnable_temps: bool = True,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        lut_impl: str = "fast",
        forward_confidence: bool = False,
        confidence_form: str = "bounded",
        confidence_gain: float = 1.0,
        z_norm: bool = False,
        bh4_block: int = 4,
        bh4_factors: int = 4,
    ):
        super().__init__()
        in_raw, out_raw = _resolve_inner(inner_dim, inner_in_dim, inner_out_dim)
        eff_in = input_dim if in_raw == -1 else in_raw
        eff_out = output_dim if out_raw == -1 else out_raw
        if eff_in < 1 or eff_out < 1:
            raise ValueError(f"effective inner dims must be >= 1 (or -1); got "
                             f"inner_in={in_raw}, inner_out={out_raw}")
        if inner_residual and eff_in != eff_out:
            raise ValueError(
                "inner_residual requires the LUT's effective input width == output width "
                f"(the skip lives in one space); got eff_in={eff_in}, eff_out={eff_out}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_in_dim = in_raw
        self.inner_out_dim = out_raw
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.has_compress = in_raw != -1
        self.has_decompress = out_raw != -1
        # legacy attribute (only meaningful when the two inner dims coincide)
        self.inner_dim = in_raw if in_raw == out_raw else None
        self.nap = nap
        self.tph = tph
        self.n_heads = n_heads
        self.inner_residual = bool(inner_residual)
        self.joint_head_compression = bool(joint_head_compression)
        self.lut_impl = lut_impl
        self.forward_confidence = bool(forward_confidence)
        self.confidence_form = confidence_form
        self.confidence_gain = float(confidence_gain)
        # Optional LayerNorm on the compressed code, applied AFTER compress and BEFORE the
        # lookup. Nothing else constrains z's scale, and because Light's address is
        # sign(d.detach()) nothing pulls it back either -- the routing margins are thresholded
        # against a code free to drift per layer, which is the depth-graded margin profile we
        # measured (Light |d| 0.00001/0.210/0.388/0.516/0.555/0.781 vs Fast flat ~0.6-0.7).
        # Normalising each head's code independently gives the margins a stable scale to live
        # on. Default OFF, so existing configs are byte-identical.
        self.z_norm_enabled = bool(z_norm)
        self.z_norm = nn.LayerNorm(eff_in, device=device) if z_norm else None
        if lut_impl not in ("fast", "light", "bh4"):
            raise ValueError(f"lut_impl must be 'fast', 'light' or 'bh4', got {lut_impl!r}")

        if lut_impl == "bh4":
            # BH4 replaces compress AND the anchor-pair addressing: a structured
            # O(d log d) transform whose output COORDINATE signs are the address, i.e.
            # LookupFFN's routing. decompress is unchanged and still sits on top, and the
            # tables keep our narrow-rows-plus-decompress layout rather than LookupFFN's
            # full-width rows. There is no compress and no z_norm on this path -- both
            # describe a code that no longer exists.
            if z_norm:
                raise ValueError("z_norm has no meaning on the bh4 path: it normalises "
                                 "the compress output, and bh4 has no compress")
            if inner_residual:
                raise ValueError("inner_residual has no meaning on the bh4 path: the "
                                 "skip would add the BH4 code to a table row")
            self.compress = nn.Identity()
            self.lut_bh4 = BH4MultiHeadLUT(
                input_dim=input_dim, n_heads=n_heads, tables_per_head=tph,
                n_anchor_pairs=nap, output_dim=eff_out, block=bh4_block,
                n_factors=bh4_factors, confidence_form=confidence_form,
                confidence_gain=confidence_gain,
                initial_weights_noise=initial_weights_noise,
                random_seed=random_seed, device=device,
            )
            self.decompress = (nn.Linear(n_heads * out_raw, output_dim, device=device)
                               if self.has_decompress else nn.Identity())
            return

        if lut_impl == "light":
            # LookupFFN-style control. The table budget n_heads*tph*2^nap*eff_out is
            # identical to the Fast path's in BOTH topologies, which is what makes the
            # ablation fair; only the projections differ, and only in the joint case.
            #
            #   independent (default, mirrors Fast's multi_head_input=True): per-head
            #     compress 384 -> n_heads*eff_in, block-diagonal routing, per-head output
            #     block, decompress n_heads*eff_out -> 384. Projections match Fast exactly.
            #   joint / no compress: one shared code, one summed ensemble, one decompress.
            mh = self.has_compress and not self.joint_head_compression and n_heads > 1
            self.light_multi_head_input = mh
            if mh:
                self.compress = nn.Linear(input_dim, n_heads * in_raw, device=device)
            else:
                self.compress = (nn.Linear(input_dim, in_raw, device=device)
                                 if self.has_compress else nn.Identity())
            self.lut_light = LightMultiHeadLUT(
                input_dim=eff_in, n_tables=n_heads * tph, output_dim=eff_out,
                n_anchor_pairs=nap, confidence_form=confidence_form,
                confidence_gain=confidence_gain,
                random_seed=random_seed, initial_weights_noise=initial_weights_noise,
                device=device, n_heads=n_heads, multi_head_input=mh,
            )
            if mh:
                self.decompress = nn.Linear(n_heads * out_raw, output_dim, device=device)
            else:
                self.decompress = (nn.Linear(out_raw, output_dim, device=device)
                                   if self.has_decompress else nn.Identity())
            return

        _lut_kw = dict(
            n_anchor_pairs=nap, tables_per_head=tph, forward_mode=forward_mode,
            backward_topk=backward_topk,
            weight_dtype=weight_dtype, use_bf16=use_bf16,
            initial_weights_noise=initial_weights_noise, learnable_temps=learnable_temps,
            device=device, forward_confidence=forward_confidence,
            confidence_form=confidence_form, confidence_gain=confidence_gain,
        )
        if self.joint_head_compression:
            # JOINT: one shared compress feeds all heads; a single FastMHL(n_heads) reads the
            # shared z; heads summed; one shared decompress.
            self.compress = (nn.Linear(input_dim, in_raw, device=device)
                             if self.has_compress else nn.Identity())
            self.lut = FastMultiHeadLut(
                input_dim=eff_in, n_heads=n_heads, n_outputs=eff_out,
                random_seed=random_seed, **_lut_kw,
            )
            self.decompress = (nn.Linear(out_raw, output_dim, device=device)
                               if self.has_decompress else nn.Identity())
        else:
            # INDEPENDENT per-head: per-head compress (row-blocks of one Linear ->
            # n_heads*inner_in), one single-head FastMHL per head (seed + h so head 0 matches
            # the joint single-head seed -> exact match at n_heads=1), and a decompress over
            # the concatenated per-head outputs (== summed per-head decompress).
            self.compress = (nn.Linear(input_dim, n_heads * in_raw, device=device)
                             if self.has_compress else nn.Identity())
            # ONE batched FastMHL either way -- there is no per-head ModuleList path any
            # more. Which batching applies depends on whether the heads have private
            # inputs, and the two are NOT interchangeable:
            #
            #   has_compress  -> multi_head_input=True. Each head owns an inner_in slice of
            #                    the compressed vector, so routing is block-diagonal: head h
            #                    reads columns [h*eff_in, (h+1)*eff_in). Anchors/weights are
            #                    drawn per head with seed (random_seed + h), which reproduces
            #                    the old ModuleList of single-head LUTs BIT-FOR-BIT.
            #
            #   no compress   -> shared input. Every head reads the same full x, which is by
            #                    definition not block-diagonal, so this is the ordinary
            #                    shared-input multi-head FastMHL (same form the joint path
            #                    uses) and it draws from `random_seed` as a single module.
            #
            # NOTE (behaviour change, deliberate): that second case previously ran through
            # the per-head loop and therefore seeded head h with (random_seed + h). Folding
            # it into one shared-input module switches it to the joint convention
            # (`random_seed`), so a model built here with inner_in_dim=-1 and
            # joint_head_compression=False no longer initialises identically to one built
            # before this change. It affects exactly one published configuration --
            # exp_n_0138 (paper section 5 / Table 3, output-compression-only) -- whose
            # reported 1.21249 bpb came from the old seeding. Forward/backward semantics,
            # shapes and parameter counts are unchanged; only the init draw differs.
            self.lut_batched = FastMultiHeadLut(
                input_dim=eff_in, n_heads=n_heads, n_outputs=eff_out,
                multi_head_input=self.has_compress, random_seed=random_seed, **_lut_kw,
            )
            self.decompress = (nn.Linear(n_heads * out_raw, output_dim, device=device)
                               if self.has_decompress else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [N, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.lut_impl == "bh4":
            # No compress: BH4 reads x directly and its coordinate signs ARE the address.
            N = x.shape[0]
            y = self.lut_bh4(x).to(x.dtype)            # [N, n_heads, eff_out]
            return self.decompress(y.reshape(N, self.n_heads * self.eff_out))

        if self.lut_impl == "light":
            N = x.shape[0]
            if self.light_multi_head_input:
                # per-head slice in, per-head block out — same shapes as the Fast path
                z = self.compress(x).view(N, self.n_heads, self.inner_in_dim)
                if self.z_norm is not None:
                    # normalises over the last axis, i.e. each head's own code, independently
                    z = self.z_norm(z)
                y = self.lut_light(z).to(z.dtype)      # [N, n_heads, eff_out]
                if self.inner_residual:
                    y = y + z
                return self.decompress(y.reshape(N, self.n_heads * self.eff_out))
            # one shared code, one summed ensemble of n_heads*tph tables, one decompress
            z = self.compress(x)                       # [N, eff_in]  (Identity -> x)
            if self.z_norm is not None:
                z = self.z_norm(z)
            y = self.lut_light(z).to(z.dtype)          # [N, eff_out]
            if self.inner_residual:
                y = y + z
            return self.decompress(y)

        if self.joint_head_compression:
            z = self.compress(x)                       # [N, eff_in]  (Identity -> x)
            y = self.lut(z).sum(dim=1).to(z.dtype)     # [N, eff_out]
            if self.inner_residual:
                y = y + z                              # eff_in == eff_out guaranteed
            return self.decompress(y)                  # [N, output_dim]  (Identity -> y)

        # INDEPENDENT per-head path -- one batched FastMHL, no per-head loop.
        N = x.shape[0]
        if self.has_compress:
            z = self.compress(x).view(N, self.n_heads, self.inner_in_dim)   # [N, H, inner_in]
        else:
            z = x                                               # [N, eff_in], shared by all heads
        y = self.lut_batched(z).to(z.dtype)                     # [N, H, eff_out]
        if self.inner_residual:
            # eff_in == eff_out guaranteed. With per-head inputs z is [N, H, eff_in] and the
            # skip is per head; with a shared input it is [N, eff_in] and the same x is added
            # to every head -- which is what the old per-head loop did (z_h = x for all h).
            y = y + (z if self.has_compress else z.unsqueeze(1))
        if self.has_decompress:
            # reshape == the old torch.cat(per-head parts, dim=-1)
            return self.decompress(y.reshape(N, self.n_heads * self.eff_out))
        return y.sum(dim=1)                                     # [N, eff_out]

    @staticmethod
    def param_count(input_dim: int, output_dim: int, inner_dim: Optional[int] = None,
                    *, inner_in_dim: Optional[int] = None, inner_out_dim: Optional[int] = None,
                    nap: int, tph: int, n_heads: int = 1,
                    joint_head_compression: bool = False) -> dict:
        """Exact parameter breakdown (dict of the three parts + total).

        A -1 inner dim drops that projection (0 params). The LUT budget uses the effective
        output width. JOINT shares one compress/decompress pair; INDEPENDENT gives each head
        its own (n_heads x the compress weight, decompress over concatenated per-head
        vectors). At n_heads=1 the two modes agree.
        """
        in_raw, out_raw = _resolve_inner(inner_dim, inner_in_dim, inner_out_dim)
        eff_out = output_dim if out_raw == -1 else out_raw
        lut = n_heads * tph * (2 ** nap) * eff_out
        if in_raw == -1:
            compress = 0
        elif joint_head_compression:
            compress = input_dim * in_raw + in_raw
        else:
            compress = input_dim * (n_heads * in_raw) + n_heads * in_raw
        if out_raw == -1:
            decompress = 0
        elif joint_head_compression:
            decompress = out_raw * output_dim + output_dim
        else:
            decompress = (n_heads * out_raw) * output_dim + output_dim
        return {
            "compress": compress,
            "lut": lut,
            "decompress": decompress,
            "total": compress + lut + decompress,
        }


# Short alias — both names refer to the same class.
CompressionMHL = CompressionMultiHeadLUT
