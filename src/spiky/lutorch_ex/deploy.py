"""Deployment baking + compact checkpoint export for lutorch_ex (PR 1).

Two independent axes (no cartridge-specific branches in projection or export):

* **Decompress-bake** (cartridge <-> ProjectionMHL): the existing ``SupportsDecompressBake`` protocol.
  For the quantised LUT the per-(group, channel) power-of-two output scale ``2^(e - 6)`` is folded,
  EXACTLY (powers of two only change the fp32 exponent field), into a copy of the decompress weight at
  export time. NOTE: the *trained* :class:`QuantisedConfidenceLUT` deliberately does NOT implement
  ``decompress_scale`` -- its live forward returns the fake-quant value (already ``* 2^e``) and
  ProjectionMHL folds on every forward, so implementing it there would double-scale. The fold is a
  deploy-only transform; :class:`DeployedQuantisedConfidenceLUT` returns the RAW int32 accumulator and
  reports ``decompress_scale() -> None`` (the fold already lives in the stored decompress weight).

* **Deployment-compaction** (general): the :class:`SupportsDeploymentExport` runtime-checkable protocol
  (``to_deployment()``). A base no-op on :class:`ManifestoLUT` passes non-quant cartridges through
  unchanged (via the normal backbone state_dict); :class:`QuantisedConfidenceLUT` overrides it to emit
  the int8 compact payload.

API::

    export_deployment(model, "deploy.lxq")                      # bake int8 + fold scale + drop fp32/exponent
    m = load_deployment("deploy.lxq", make_skeleton, device="cuda").eval()   # fp32 LUT table NEVER allocated
    y = m(x)

``make_skeleton`` builds the architecture on the ``meta`` device (no real allocation); load swaps the
ProjectionMHL FFNs for :class:`DeployedQuantisedConfidenceLUT` (real int8) and materialises the backbone
from the stored state_dict.

Serialisation: safetensors when available, else a ``weights_only=True`` torch load (pickle-free on load);
either way only tensors + a small JSON-able meta dict, no arbitrary pickled objects executed on load.
"""
from __future__ import annotations

import json
from typing import Callable, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from .lut_spec import LUTSpec
from .projection import ProjectionMHL
from .cartridges import _pow2
from .cartridges._fused_ops import _global_cells
from .cartridges.quantised_confidence import QuantisedConfidenceLUT

_FORMAT_VERSION = 1


@runtime_checkable
class SupportsDeploymentExport(Protocol):
    """Capability: a cartridge that can emit a compact deployment payload via ``to_deployment()``.

    Returns ``{"format": tag, "decompress_scale"?: Tensor, "tensors": {name: Tensor}, "meta": {...}}``.
    ``format == "dense"`` means "no compaction; serialise my params via the normal state_dict".
    """

    def to_deployment(self) -> dict:
        ...


# ---- the inference-only deployed quantised cartridge -------------------------------------------

class DeployedQuantisedConfidenceLUT(QuantisedConfidenceLUT):
    """Inference-only p2_int8 cartridge: holds the int8 ``packed`` table (NO fp32 master Parameter),
    the anchors and the β/γ/τ scalars as buffers. Reuses the base addressing (which needs only
    spec/anchors/powers) and returns the RAW int32 accumulator (units 2^-6); the per-(group,channel)
    output scale lives, pre-folded, in the ProjectionMHL decompress weight. read_top_n==2 only."""

    def __init__(self, spec: LUTSpec, tensors: dict, meta: dict, device=None):
        nn.Module.__init__(self)   # skip the fp32-weight-allocating ManifestoLUT.__init__
        self.spec = spec
        self.read_top_n = int(meta["read_top_n"])
        if self.read_top_n != 2:
            raise NotImplementedError("DeployedQuantisedConfidenceLUT supports read_top_n==2 only")
        self.single = (spec.anchor_mode == "single")
        self.cmp_eps = float(meta.get("cmp_eps", 0.0))
        self._quant = _pow2.resolve_quant_config(meta.get("quant_mode", "p2_int8"))
        self.table_dropout_rate = 0.0
        self._compiled = None
        self._compiled_train = None
        self._compiled_addr = None
        dev = torch.device(device) if device is not None else None
        self.register_buffer("packed", tensors["packed"].to(dev) if dev else tensors["packed"])  # int8 [n_tables,K,d_out]
        for b in ("anchor_a", "anchor_b", "powers", "in_head", "out_head",
                  "confidence_log_beta", "confidence_log_gamma", "log_read_tau"):
            t = tensors[b]
            self.register_buffer(b, t.to(dev) if dev else t)

    def decompress_scale(self) -> Optional[torch.Tensor]:
        return None   # the pow2 output scale is already folded (statically) into the stored decompress weight

    def _supports_low_precision(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # inference; fp32/fp64
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self._quant
        spec = self.spec
        G, tph, K, d_out = spec.n_groups, spec.tph, spec.n_cells, spec.d_out
        B = x.shape[0]
        z, u, c, j_star, u_abs_star, c_alt = self._addresses(x)
        beta, gamma = self._betagamma(u.dtype)
        g0 = torch.zeros((), dtype=u.dtype, device=u.device)
        tau = self.log_read_tau.to(u.dtype).exp()
        m = u.abs(); mv = u_abs_star.unsqueeze(-1)
        q, k, skip, drop = _pow2.blend_exponents(m, mv, tau, g0, beta, gamma, cfg)
        gc = _global_cells(c, G, tph, K); gca = _global_cells(c_alt, G, tph, K)
        flat_idx = torch.stack([gc, gca], dim=-1).reshape(B, G, tph, 2)
        packed = self.packed.reshape(G * tph * K, d_out)
        acc = _pow2.int_blend_read(packed, cfg["bits"], d_out, flat_idx, q, k, skip, drop)  # int32 [B,G,d_out]
        return self._route(acc.to(x.dtype), x)   # RAW (units 2^-6); scale folded into decompress


# ---- (de)serialisation (pickle-free on load) ---------------------------------------------------

def _save_payload(path: str, tensors: dict, meta: dict) -> None:
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, path,
                  metadata={"lxq_meta": json.dumps(meta)})
    except ImportError:
        torch.save({"tensors": {k: v.contiguous().cpu() for k, v in tensors.items()},
                    "meta_json": json.dumps(meta)}, path)


def _load_payload(path: str):
    try:
        from safetensors import safe_open
        tensors, meta = {}, None
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["lxq_meta"])
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors, meta
    except (ImportError, Exception):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        return obj["tensors"], json.loads(obj["meta_json"])


# ---- export / load -----------------------------------------------------------------------------

_BUFKEYS = ("confidence_log_beta", "confidence_log_gamma", "log_read_tau",
            "anchor_a", "anchor_b", "powers", "in_head", "out_head")


def export_deployment(model: nn.Module, path: str) -> dict:
    """Walk ``model``; bake every ProjectionMHL whose cartridge compactifies (``to_deployment`` with a
    non-"dense" format) to int8 + a decompress weight with the pow2 output scale folded in (exact);
    serialise those plus the remaining (backbone) params. Drops the fp32 LUT table and the raw
    exponent buffer. Returns the meta dict. Training is untouched (export reads, never mutates)."""
    meta = {"format_version": _FORMAT_VERSION, "ffns": {}}
    tensors: dict = {}
    compacted: list[str] = []
    for name, mod in model.named_modules():
        if not (isinstance(mod, ProjectionMHL) and isinstance(mod.cartridge, SupportsDeploymentExport)):
            continue
        payload = mod.cartridge.to_deployment()
        if payload.get("format") == "dense":
            continue   # non-quant cartridge: handled by the backbone state_dict, unchanged
        if not mod.has_decompress:
            raise ValueError(f"{name}: deployment export needs a decompress Linear to fold the pow2 scale into")
        scale = payload["decompress_scale"]                      # [out_features]
        dec_w = (mod.decompress.weight.detach().cpu() * scale.reshape(1, -1).cpu()).contiguous()  # EXACT pow2 fold
        for key, t in payload["tensors"].items():
            tensors[f"{name}::{key}"] = t
        tensors[f"{name}::compress.weight"] = mod.compress.weight.detach()
        if mod.compress.bias is not None:
            tensors[f"{name}::compress.bias"] = mod.compress.bias.detach()
        tensors[f"{name}::decompress.weight"] = dec_w
        if mod.decompress.bias is not None:
            tensors[f"{name}::decompress.bias"] = mod.decompress.bias.detach()
        meta["ffns"][name] = {**payload["meta"], "d_model": mod.d_model,
                              "has_compress_bias": mod.compress.bias is not None,
                              "has_decompress_bias": mod.decompress.bias is not None}
        compacted.append(name)
    # backbone: everything NOT inside a compacted FFN (the compacted FFNs are rebuilt from the int8 payload)
    for n, p in model.state_dict().items():
        if any(n == m or n.startswith(m + ".") for m in compacted):
            continue
        tensors[f"__backbone__::{n}"] = p.detach()
    _save_payload(path, tensors, meta)
    return meta


def _set_submodule(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() and not hasattr(parent, p) else getattr(parent, p)
    last = parts[-1]
    if last.isdigit() and not hasattr(parent, last):
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def load_deployment(path: str, make_skeleton: Callable[[], nn.Module], device=None) -> nn.Module:
    """Rebuild an inference-only model from a compact checkpoint WITHOUT allocating any fp32 LUT table.

    ``make_skeleton()`` must build the architecture on the ``meta`` device (no real memory). Each
    ProjectionMHL FFN that was compacted is replaced by a real :class:`DeployedQuantisedConfidenceLUT`
    (int8) wrapped in a ProjectionMHL whose decompress weight already has the pow2 scale folded in; the
    remaining (backbone) params are materialised from the stored state_dict."""
    tensors, meta = _load_payload(path)
    model = make_skeleton()
    for name, ff in meta["ffns"].items():
        spec = LUTSpec(h_in=ff["spec"]["h_in"], h_out=ff["spec"]["h_out"], tph=ff["spec"]["tph"],
                       nap=ff["spec"]["nap"], d_in=ff["spec"]["d_in"], d_out=ff["spec"]["d_out"])
        cart_tensors = {b: tensors[f"{name}::{b}"] for b in ("packed",) + _BUFKEYS}
        dep = DeployedQuantisedConfidenceLUT(spec, cart_tensors, ff, device=device)
        proj = ProjectionMHL(dep, d_model=ff["d_model"], bias=ff["has_decompress_bias"], device=device)
        with torch.no_grad():
            proj.compress.weight.copy_(tensors[f"{name}::compress.weight"].to(proj.compress.weight.device))
            if ff["has_compress_bias"]:
                proj.compress.bias.copy_(tensors[f"{name}::compress.bias"].to(proj.compress.bias.device))
            proj.decompress.weight.copy_(tensors[f"{name}::decompress.weight"].to(proj.decompress.weight.device))
            if ff["has_decompress_bias"]:
                proj.decompress.bias.copy_(tensors[f"{name}::decompress.bias"].to(proj.decompress.bias.device))
        _set_submodule(model, name, proj)
    backbone = {n[len("__backbone__::"):]: t for n, t in tensors.items() if n.startswith("__backbone__::")}
    if backbone:
        dev = torch.device(device) if device is not None else torch.device("cpu")
        model.load_state_dict({k: v.to(dev) for k, v in backbone.items()}, strict=False, assign=True)
    return model.eval()
