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

``make_skeleton`` builds the architecture on the ``meta`` device (no real allocation); load swaps each
compacted ProjectionMHL FFN for the inference cartridge its stored format tag names (via the
:mod:`.deploy_registry` ``format -> from_deployment`` registry; e.g. ``"p2_int8"`` ->
``DeployedQuantisedConfidenceLUT``) and materialises the backbone from the stored state_dict. This
module itself names no concrete cartridge class and holds no int8 knowledge.

Serialisation: safetensors when it can write here (it needs numpy), else ``torch.save``. The loader picks
the reader from the file's format, so either file loads anywhere it can be read; the torch format is read
with ``weights_only=True`` (pickle-free on load). Either way only tensors + a small JSON-able meta dict, no
arbitrary pickled objects executed on load.
"""
from __future__ import annotations

import json
from typing import Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn

from .lut_spec import LUTSpec
from .projection import ProjectionMHL
from .deploy_registry import get_rebuilder

_FORMAT_VERSION = 1


@runtime_checkable
class SupportsDeploymentExport(Protocol):
    """Capability: a cartridge that can emit a compact deployment payload via ``to_deployment()``.

    Returns ``{"format": tag, "decompress_scale"?: Tensor, "tensors": {name: Tensor}, "meta": {...}}``.
    ``format == "dense"`` means "no compaction; serialise my params via the normal state_dict".
    """

    def to_deployment(self) -> dict:
        ...


# ---- (de)serialisation (pickle-free on load) ---------------------------------------------------

_ZIP_MAGIC = b"PK\x03\x04"    # torch.save files are zip archives; safetensors files start with a header length


def _save_payload(path: str, tensors: dict, meta: dict) -> None:
    tensors = {k: v.contiguous().cpu() for k, v in tensors.items()}
    try:
        from safetensors.torch import save_file
        save_file(tensors, path, metadata={"lxq_meta": json.dumps(meta)})
        return
    except ImportError:     # safetensors is not installed, or numpy, which safetensors.torch needs to write, is not
        pass
    torch.save({"tensors": tensors, "meta_json": json.dumps(meta)}, path)


def _load_payload(path: str):
    # The reader follows the file's own format, not what is installed here: a file written by either writer
    # loads in any environment that can read that format.
    with open(path, "rb") as f:
        is_torch_file = f.read(4) == _ZIP_MAGIC
    if is_torch_file:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        return obj["tensors"], json.loads(obj["meta_json"])
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(f"{path} is a safetensors file; install safetensors to load it") from e
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["lxq_meta"])
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


# ---- export / load -----------------------------------------------------------------------------


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
        # input_dim / output_dim are the wrapper's widths (they may differ); "d_model" is kept for
        # readers of older files and is only meaningful (and only written) when the two are equal.
        dims = {"input_dim": mod.input_dim, "output_dim": mod.output_dim}
        if mod.input_dim == mod.output_dim:
            dims["d_model"] = mod.input_dim
        meta["ffns"][name] = {**payload["meta"], "format": payload["format"], **dims,
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
    ProjectionMHL FFN that was compacted is replaced by a real inference cartridge, rebuilt via the
    format-tag registry (``meta["format"]`` -> the cartridge's ``from_deployment``), wrapped in a
    ProjectionMHL whose decompress weight already has the pow2 scale folded in; the remaining
    (backbone) params are materialised from the stored state_dict. This module names no concrete
    cartridge class: which inference cartridge to build is decided entirely by the stored format tag."""
    tensors, meta = _load_payload(path)
    model = make_skeleton()
    for name, ff in meta["ffns"].items():
        spec = LUTSpec(h_in=ff["spec"]["h_in"], h_out=ff["spec"]["h_out"], tph=ff["spec"]["tph"],
                       nap=ff["spec"]["nap"], d_in=ff["spec"]["d_in"], d_out=ff["spec"]["d_out"])
        # Collect this FFN's cartridge tensors by prefix (everything under "{name}::" except the
        # ProjectionMHL compress/decompress weights); the cartridge's from_deployment knows its keys.
        prefix = f"{name}::"
        cart_tensors = {k[len(prefix):]: t for k, t in tensors.items()
                        if k.startswith(prefix)
                        and not k.startswith(prefix + "compress.")
                        and not k.startswith(prefix + "decompress.")}
        dep = get_rebuilder(ff["format"])(spec, cart_tensors, ff, device=device)
        # Files written before input_dim/output_dim existed carry only "d_model" (square wrappers).
        in_dim = ff.get("input_dim", ff.get("d_model"))
        out_dim = ff.get("output_dim", ff.get("d_model"))
        proj = ProjectionMHL(dep, input_dim=in_dim, output_dim=out_dim,
                             bias=ff["has_decompress_bias"], device=device)
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
