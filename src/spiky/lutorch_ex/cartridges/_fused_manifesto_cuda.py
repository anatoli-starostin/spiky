"""The CUDA implementation of FusedManifestoHardLUT / FusedManifestoSoftLUT, their only one: hand-written
forward + backward kernels (``csrc/fused_manifesto.cu``), built the way FusedConfidenceLUT's are. A case the kernels do
not cover, or a missing extension, raises; the reference classes (ManifestoHardLUT / ManifestoSoftLUT) cover the rest.

One forward and one backward kernel per step run the whole cartridge: addressing (margins, MSB-first address,
least-confident bit, neighbour), the neighbour weight w (``weight="rational"``: 0.5 / (1 + |u*|); ``"softsign"``: the
learned-temperature sigmoid, the two temperatures passed in as a device tensor), the (weighted) read summed over the
group's tables, and on the backward the weight-gradient scatter (vector float32 atomics), the j*-pair margin gradient
into z and, for ``softsign``, the two temperatures' gradients (summed per CTA, then one atomic add each).
Nothing per table goes through global memory (no ``[B, G*tph, d_out]`` expanded gradient, no materialised addressing
tensors); the backward recomputes the addressing from z.

Hard and Soft share the kernels and differ only in the read weights (see the .cu header for the math):
  HARD  value  m W[c]                 table grad on c only        (straight-through: the input / temperature
                                                                   gradient is that of the (1-w)/w blend)
  SOFT  value  m ((1-w) W[c] + w W[c'])  table grad on c and c'   (the exact derivative of the blend)
with the table-dropout factor m.
"""
from __future__ import annotations

import torch

from . import _fallback
from ._cuda_ext import (_MAX_CTA_THREADS, Extension, ManifestoCudaKnobs, _launch, _row_launch,  # noqa: F401
                        device_problem, forward_group_major, not_covered, row_problem)
from ._dtypes import _CUDA_INPUT_DTYPES, _CUDA_TABLE_DTYPES, unsupported_dtype

MODE_HARD, MODE_SOFT = 0, 1
MAX_NAP = 16
EXT = Extension("lutorch_ex_fused_manifesto", ["fused_manifesto.cu"])


REFERENCE = "ManifestoHardLUT / ManifestoSoftLUT"


def require_cuda(module, x: torch.Tensor) -> None:
    """Raise unless the kernels can run this input: a RuntimeError naming every case they do not cover (no CUDA
    device or a CPU input, an input other than float32 / bfloat16 / float16, a table other than float32 / bfloat16,
    nap > 16, d_in beyond int16 anchors, a d_out one CTA cannot cover), or the classified NativeUnavailableError when
    only the extension is missing."""
    problems = [p for p in (device_problem(x),) if p]
    if x.dtype not in _CUDA_INPUT_DTYPES:
        problems.append(unsupported_dtype("input", x.dtype, _CUDA_INPUT_DTYPES))
    if module.weights.dtype not in _CUDA_TABLE_DTYPES:
        problems.append(unsupported_dtype("table", module.weights.dtype, _CUDA_TABLE_DTYPES))
    if module.spec.nap > MAX_NAP:
        problems.append(f"nap={module.spec.nap} (max {MAX_NAP}: the cell index packs at most {MAX_NAP} bits)")
    if module.spec.d_in > torch.iinfo(torch.int16).max:
        problems.append(f"d_in={module.spec.d_in} (max {torch.iinfo(torch.int16).max}: anchors are int16)")
    problems += [p for p in (row_problem(module.spec.d_out, *_row_launch(module.knobs, module.spec.d_out,
                                                                          module.weights.dtype)),) if p]
    if problems:
        raise RuntimeError(not_covered(type(module).__name__, problems, REFERENCE))
    if EXT.get() is None:
        _fallback.raise_unavailable(type(module).__name__, EXT.name)


class _FusedManifesto(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weights, temps, anc_a, anc_b, mask, nap, mode, knobs, out_bf16=False):
        # z: float32 or bfloat16 [B, G, d_in]. weights: float32 or bfloat16 [G, tph, K, d_out] (read in place).
        # temps: float32 [2] (T_soft, T_select) for weight="softsign", None for "rational". mask: float32 [B, G, tph]
        # or None. out_bf16: the kernel stores its (float32-accumulated) output as bfloat16; only for a bfloat16 z.
        ext = EXT.get()
        W2 = weights.reshape(-1, weights.shape[-1])
        fwd_threads, bwd_threads, vec = _launch(knobs, W2.shape[1], weights.dtype)
        out = ext.manifesto_fwd(z, anc_a, anc_b, W2, mask, temps, nap, mode, fwd_threads, knobs.rows_per_cta,
                                vec, out_bf16, forward_group_major("MANI", z.device))
        ctx.save_for_backward(z, weights, temps, anc_a, anc_b, mask)
        ctx.cfg = (nap, mode, knobs, vec, bwd_threads)
        return out                                                   # float32 [B, G, d_out]

    @staticmethod
    def backward(ctx, go):
        z, weights, temps, anc_a, anc_b, mask = ctx.saved_tensors
        nap, mode, knobs, vec, bwd_threads = ctx.cfg
        W2 = weights.reshape(-1, weights.shape[-1])
        gW, gz, gtemps = EXT.get().manifesto_bwd(
            go.float().contiguous(), z, anc_a, anc_b, W2, mask, temps, nap, mode,
            bwd_threads, knobs.rows_per_cta, vec, knobs.vec_atomics)
        # gW accumulates in float32 whatever the table dtype; cast once to the parameter's dtype.
        # gz comes back in z's dtype (a bfloat16 z: rounded once, in the kernel's store); the .to is then a no-op.
        gt = gtemps if temps is not None and ctx.needs_input_grad[2] else None
        return (gz.to(z.dtype), gW.view_as(weights).to(weights.dtype), gt,
                None, None, None, None, None, None, None)


def init_cuda(module, knobs) -> None:
    """Per-instance setup: launch knobs and int16 copies of the anchors (non-persistent buffers: they follow ``.to()`` /
    ``.cuda()`` and stay out of the state_dict; the int64 buffers remain the source of truth)."""
    module.knobs = knobs if knobs is not None else ManifestoCudaKnobs()
    if module.spec.d_in <= torch.iinfo(torch.int16).max:
        module.register_buffer("_anc_a16", module.anchor_a.to(torch.int16).contiguous(), persistent=False)
        if module.anchor_b is not None:
            module.register_buffer("_anc_b16", module.anchor_b.to(torch.int16).contiguous(), persistent=False)
        else:
            module._anc_b16 = None


def manifesto_cuda_forward(module, x: torch.Tensor, mode: int) -> torch.Tensor:
    """The fused forward: ``[B, h_in, d_in]`` -> ``[B, h_out, d_out]`` in the input dtype. Raises on a case the kernels
    do not cover or a missing extension (:func:`require_cuda`)."""
    module._check_input(x)
    require_cuda(module, x)
    # Group g reads input head g % h_in. When h_in == n_groups that map is the identity, so x already IS the per-group
    # input: skip the gather (eager it is a full extra pass + intermediate, which the compiled path used to fuse away).
    # A gather (h_in != n_groups) can send one input head to several groups: cast a low-precision input to float32
    # BEFORE it, so those groups' input gradients sum in float32 and are rounded to the input dtype once.
    zin = x if module.spec.h_in == module.spec.n_groups else \
        (x if x.dtype == torch.float32 else x.float())[:, module.in_head, :]
    # A bfloat16 input goes to the kernels as is: they upconvert it while staging each row (float32 addressing, no
    # float32 copy of the input in global memory). float16 / float64 are cast to float32 here as before.
    z = (zin if zin.dtype in (torch.float32, torch.bfloat16) else zin.float()).contiguous()
    # Table dropout: the cartridge's own mask (one source of truth with the reference classes: same draw,
    # and an overridden _table_dropout_mask is honoured), float32, applied per table to the value and every gradient.
    mask = module._table_dropout_mask(x.shape[0], x.device, torch.float32)
    mask = None if mask is None else mask.contiguous()
    # Routing gate: the kernel may store bfloat16 directly only when _route is the identity (h_out == n_groups: per-head
    # and fan-out). Fan-in sums groups in _route, which must happen on the float32 output before the single cast below.
    out_bf16 = z.dtype == torch.bfloat16 and module.spec.h_out == module.spec.n_groups
    # weight="softsign": the two temperatures, float32, on the device (no host sync); autograd carries their gradient
    # back to the log-parametrised leaves.
    temps = torch.stack(module._softsign_temps(torch.float32)).contiguous() if module.weight == "softsign" else None
    grp = _FusedManifesto.apply(z, module.weights, temps, module._anc_a16, module._anc_b16, mask,
                                module.spec.nap, mode, module.knobs, out_bf16)
    return module._route(grp, x).to(x.dtype)
