"""QuantisedLightFFN: the exported, frozen inference form of a quant_mode light CompressionMultiHeadLUT layer.

Built only by CompressionMultiHeadLUT.export_quantised(); never used for training and never loaded back into a training
model (the training checkpoint stays the one source of the float master weights).

Contents, all buffers:
  compress_weight / compress_bias   copy of compress (E -> H * D_in)
  anchor_a / anchor_b / powers / table_offset
  tables                            packed int-b rows [H * T * 2^NAP, D'] as written by pow2_read.pack_tables for
                                    meta["bits"] (only int8 exists: int8 [.., D])
  tau, g, beta, gamma               snapshots of the blend temperature and the learned_margin scalars
  decompress_weight / _bias         decompress with column h * D + c multiplied by 2^(e[h, c] - 6)
                                    (the per-(head, channel) weight scale and the fixed-point unit, folded)
and a `meta` dict (preset constants incl. the explicit table width `bits` and weight `offset`, and geometry).

Forward: compress, anchor margins and cell addresses, the per-table integers (c1, c2, q, k', skip, drop), the int32 shift-add
over the int8 rows (note Section 6), then the folded decompress. Two implementations of that one read, bit-identical:

  * CUDA fp32 input with the pow2_int8_cuda extension available (RTX 5090): the fused kernel -- the per-table integers
    (p2::table_scalars, the same function the training forward's spiky_lutorch::p2_scalars op runs) and the int8
    accumulation in ONE launch. Always used when available.
  * otherwise (CPU, another GPU, no extension, SPIKY_P2_CUDA_DISABLE=1): the torch read, _forward_torch -- the integers from
    pow2_scalar_op.table_integers and pow2_read.int_blend_read, compiled on CUDA.

File format (to_file / from_file): torch.save of {"format": FORMAT, "version": VERSION, "meta": ..., "buffers": ...}.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pow2_int8_cuda
from . import pow2_read
from . import pow2_scalar_op

FORMAT = "spiky.lutorch.QuantisedLightFFN"
VERSION = 1
_BUFFERS = ("compress_weight", "compress_bias", "anchor_a", "anchor_b", "powers", "table_offset", "tables",
            "tau", "g", "beta", "gamma", "decompress_weight", "decompress_bias")


class QuantisedLightFFN(nn.Module):
    def __init__(self, meta: dict, buffers: dict):
        super().__init__()
        missing = [k for k in _BUFFERS if k not in buffers]
        if missing:
            raise ValueError(f"QuantisedLightFFN is missing buffers {missing}")
        if meta.get("bits") != 8:
            raise NotImplementedError(f"QuantisedLightFFN reads int8 tables only; this artefact has bits={meta.get('bits')!r}")
        self.meta = dict(meta)
        for k in _BUFFERS:
            self.register_buffer(k, buffers[k])
        self.cfg = {k: meta[k] for k in ("mode", "bits", "offset", "Q", "L", "kmax", "lo", "hi", "J", "C")}
        self.requires_grad_(False)
        self._compiled = None
        self._compile_enabled = os.environ.get("LUT_DISABLE_COMPILE") != "1"
        self._kcache = None
        pow2_scalar_op.ensure_registered()          # eager: build/register the CUDA op before any compiled forward

    @classmethod
    @torch.no_grad()
    def from_training(cls, ffn) -> "QuantisedLightFFN":
        lut = ffn.lut_light
        cfg = lut._quant
        H, T, D, NAP = lut.n_heads, lut.tables_per_head, lut.output_dim, lut.n_anchor_pairs
        e, packed, bits = lut.quantised_tables()
        tau, g, beta, gamma = lut._quant_scalars()
        W_dec = ffn.decompress.weight.detach().clone()                          # [E, H * D]
        fold = torch.pow(2.0, e.to(W_dec.dtype) - pow2_read.FIXED_POINT_SHIFT).reshape(1, H * D)
        meta = dict(cfg, n_heads=H, tables_per_head=T, output_dim=D, input_dim=lut.input_dim, n_anchor_pairs=NAP,
                    table_size=lut.table_size, model_dim=ffn.input_dim, format=FORMAT, version=VERSION)
        buffers = dict(
            compress_weight=ffn.compress.weight.detach().clone(), compress_bias=ffn.compress.bias.detach().clone(),
            anchor_a=lut.anchor_a.detach().clone(), anchor_b=lut.anchor_b.detach().clone(),
            powers=lut.powers.detach().clone(), table_offset=lut.table_offset.detach().clone(),
            tables=packed.clone(),
            tau=tau.detach().clone(), g=g.detach().clone(), beta=beta.detach().clone(), gamma=gamma.detach().clone(),
            decompress_weight=W_dec * fold, decompress_bias=ffn.decompress.bias.detach().clone())
        return cls(meta, buffers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eval only: the fused CUDA kernel when it can serve this input, else the torch read (same output, bit for bit)."""
        if x.dim() != 2 or x.shape[1] != self.meta["model_dim"]:
            raise ValueError(f"x must be [N, {self.meta['model_dim']}], got {tuple(x.shape)}")
        with torch.no_grad():
            if self._uses_kernel(x):
                return self._forward_fused(x)
            return self._forward_torch(x)

    def _uses_kernel(self, x: torch.Tensor) -> bool:
        return x.is_cuda and x.dtype == torch.float32 and pow2_int8_cuda.load() is not None

    # ------------------------------------------------------------------ fused CUDA kernel -------------------------------
    def _kernel_cache(self, device):
        """The kernel's view of the artefact on `device`: stride-padded int8 rows, int32 anchors, one-element fp32 scalars."""
        if self._kcache is None or self._kcache["device"] != device:
            self._kcache = dict(
                device=device, tables=pow2_int8_cuda.stride_tables(self.tables.to(device), self.meta["output_dim"]),
                anchor_a=self.anchor_a.to(device=device, dtype=torch.int32).contiguous(),
                anchor_b=self.anchor_b.to(device=device, dtype=torch.int32).contiguous(),
                scalars=tuple(t.to(device=device, dtype=torch.float32).reshape(1).contiguous()
                              for t in (self.tau, self.g, self.beta, self.gamma)))
        return self._kcache

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        mt = self.meta
        H, NAP, Din, D = mt["n_heads"], mt["n_anchor_pairs"], mt["input_dim"], mt["output_dim"]
        N = x.shape[0]
        kc = self._kernel_cache(x.device)
        z = F.linear(x, self.compress_weight, self.compress_bias).view(N, H, Din)
        acc = pow2_int8_cuda.read_fused(z, kc["anchor_a"], kc["anchor_b"], kc["tables"], kc["scalars"], NAP, D,
                                        self.cfg["lo"], self.cfg["hi"], self.cfg["Q"])
        return F.linear(acc.reshape(N, H * D), self.decompress_weight, self.decompress_bias)

    # ------------------------------------------------------------------ torch read ------------------------------------
    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """The torch read: compiled on CUDA (one generated kernel over the int8 bytes); eager with the bags chunked elsewhere,
        with LUT_DISABLE_COMPILE=1, or if compilation fails."""
        if x.is_cuda and self._compile_enabled:
            if self._compiled is None:
                try:
                    self._compiled = torch.compile(self._forward_impl, dynamic=True)
                except Exception:
                    self._compile_enabled = False
                    return self._forward_impl(x, 4096)
            return self._compiled(x, None)
        return self._forward_impl(x, 4096)

    def _margins(self, x: torch.Tensor):
        mt = self.meta
        H, T, NAP, Din = mt["n_heads"], mt["tables_per_head"], mt["n_anchor_pairs"], mt["input_dim"]
        N = x.shape[0]
        z = F.linear(x, self.compress_weight.to(x.dtype), self.compress_bias.to(x.dtype)).view(N, H, Din)
        idx_a = self.anchor_a.reshape(1, H, T * NAP).expand(N, H, T * NAP)
        idx_b = self.anchor_b.reshape(1, H, T * NAP).expand(N, H, T * NAP)
        d = (torch.gather(z, 2, idx_a) - torch.gather(z, 2, idx_b)).view(N, H, T, NAP)
        index = ((d > 0).to(torch.int64) * self.powers.view(1, 1, 1, -1)).sum(dim=-1)
        return d, index

    def _scalars(self, dtype):
        return tuple(t.to(dtype) for t in (self.tau, self.g, self.beta, self.gamma))

    def _forward_impl(self, x: torch.Tensor, chunk_bags):
        mt = self.meta
        H, T, D = mt["n_heads"], mt["tables_per_head"], mt["output_dim"]
        N = x.shape[0]
        d, index = self._margins(x)
        idx, q, k, skip, drop = pow2_scalar_op.table_integers(d, index, self.powers, *self._scalars(x.dtype), self.cfg)
        acc = pow2_read.int_blend_read(self.tables, self.cfg["bits"], D, idx + self.table_offset.view(1, H, T, 1),
                                       q, k, skip, drop, chunk_bags)
        y = acc.to(x.dtype).reshape(N, H * D)
        return F.linear(y, self.decompress_weight.to(x.dtype), self.decompress_bias.to(x.dtype))

    def _reference_cells(self, x: torch.Tensor) -> torch.Tensor:
        """TEST ORACLE: the packed per-table integers uint8 [N, H, T, 3] for pow2_int8_cuda.read_cells, computed outside the
        fused kernel (by the p2_scalars op when available, else pow2_read)."""
        d, index = self._margins(x)
        return pow2_scalar_op.table_cells(d, index, self.powers, *self._scalars(x.dtype), self.cfg)

    def to_file(self, path: str) -> None:
        torch.save({"format": FORMAT, "version": VERSION, "meta": self.meta,
                    "buffers": {k: getattr(self, k).detach().cpu() for k in _BUFFERS}}, path)

    @classmethod
    def from_file(cls, path: str, device=None) -> "QuantisedLightFFN":
        blob = torch.load(path, map_location=device or "cpu", weights_only=False)
        if blob.get("format") != FORMAT or blob.get("version") != VERSION:
            raise ValueError(f"not a {FORMAT} v{VERSION} file: format={blob.get('format')!r} version={blob.get('version')!r}")
        return cls(blob["meta"], blob["buffers"])

    def table_bytes(self) -> int:
        return self.tables.numel() * self.tables.element_size()

    def extra_repr(self) -> str:
        mt = self.meta
        return (f"mode={mt['mode']}, H={mt['n_heads']}, T={mt['tables_per_head']}, NAP={mt['n_anchor_pairs']}, "
                f"D={mt['output_dim']}, Q={mt['Q']}, window=[{mt['lo']}, {mt['hi']}], table_bytes={self.table_bytes()}")
