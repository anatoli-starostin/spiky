"""External subclass adding a multihead (no-collapse) output mode to
CompressionMultiHeadLUT — WITHOUT modifying the shared src/spiky/lutorch/
compression_mhl.py (standing rule: shared lutorch modules are extended by
subclassing/wrapping externally only, never edited in place).

`CompressionMultiHeadLUTMH(..., multihead_output=True)` returns the per-head
tensor [N, n_heads, eff_out] WITHOUT summing/decompressing the head axis, so the
module can drive multihead attention (separate per-head q/k/v). This is the
"Option A" from the design review. Default multihead_output=False reproduces the
stock CompressionMultiHeadLUT bit-for-bit (it just delegates to super().forward).

Constraints (validated in __init__):
  * multihead_output=True requires inner_out_dim == -1 (has_decompress=False):
    a decompress Linear mixes heads, which is incompatible with keeping them
    separate.
  * joint_head_compression=True is not supported with multihead_output=True
    (the joint path already sums the head axis before decompress); raises.
  * has_compress must be True (each head needs its own compressed slice);
    inner_in_dim == -1 would feed every head the same x, which is not a
    meaningful multihead attention projection — raises.
"""
import torch
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT


class CompressionMultiHeadLUTMH(CompressionMultiHeadLUT):
    def __init__(self, *args, multihead_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihead_output = bool(multihead_output)
        if self.multihead_output:
            if self.has_decompress:
                raise ValueError(
                    "multihead_output=True requires inner_out_dim == -1 (no decompress): "
                    "a decompress Linear mixes the head axis, incompatible with keeping "
                    f"heads separate. Got inner_out_dim={self.inner_out_dim!r}.")
            if self.joint_head_compression:
                raise ValueError(
                    "multihead_output=True is not supported with joint_head_compression=True "
                    "(the joint path sums heads before decompress).")
            if not self.has_compress:
                raise ValueError(
                    "multihead_output=True requires a compress projection (inner_in_dim != -1); "
                    "with no compress every head reads the same x.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Default: exact stock behavior (heads collapsed) — bit-for-bit unchanged.
        if not self.multihead_output:
            return super().forward(x)

        # multihead path: mirror the INDEPENDENT per-head path of the parent but
        # RETURN [N, n_heads, eff_out] instead of collapsing the head axis.
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x shape must be [N, {self.input_dim}], got {tuple(x.shape)}")
        N = x.shape[0]
        z = self.compress(x).view(N, self.n_heads, self.inner_in_dim)   # [N, H, inner_in]
        if self.batched_multi_head_input:
            z3 = z
            if self.pre_lut_meanabsnorm:
                z3 = z3 / (z3.abs().mean(-1, keepdim=True) + 1e-6)
            y = self.lut_batched(z3).to(z3.dtype)                       # [N, H, eff_out]
            if self.inner_residual:
                y = y + z3
            return y
        parts = []
        for h, lut in enumerate(self.luts):
            z_h = z[:, h, :]
            if self.pre_lut_meanabsnorm:
                z_h = z_h / (z_h.abs().mean(-1, keepdim=True) + 1e-6)
            y_h = lut(z_h).sum(dim=1).to(z_h.dtype)                     # [N, eff_out]
            if self.inner_residual:
                y_h = y_h + z_h
            parts.append(y_h)
        return torch.stack(parts, dim=1)                                # [N, H, eff_out]
