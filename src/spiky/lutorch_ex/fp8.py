"""fp8 GEMMs for ProjectionMHL's compress / decompress projections (opt-in; see ProjectionMHL ``fp8_projections``).

A self-contained copy of the tensorwise recipe nanochat trains its backbone with (nanochat/fp8.py), so lutorch_ex does
not depend on nanochat:

* the fp32 master weight is kept; every call casts BOTH GEMM operands just-in-time to ``float8_e4m3fn`` with one
  amax-derived scale per tensor ("tensorwise"), and the matmul runs as ``torch._scaled_mm`` (cuBLAS fp8 kernel);
* the forward accumulates in fp32 (``use_fast_accum=False``) and returns ``out_dtype`` (fp32 or bf16);
* the backward casts the incoming gradient to ``float8_e5m2`` (wider range) and runs the two gradient GEMMs in fp8 too;
  ``grad_weight`` is produced in fp32 for the fp32 master, ``grad_input`` in the input's dtype.

Only the GEMM operands are ever fp8; nothing fp8 is stored as a parameter. fp8 needs a CUDA device whose torch build
runs ``torch._scaled_mm`` (Ada / Hopper / Blackwell); :func:`fp8_available` probes that functionally (never by name).
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

FP8_PROJECTIONS = ("compress", "decompress")
_EPS = 1e-12


@lru_cache(maxsize=None)
def _probe(device_index: int) -> tuple[bool, str]:
    try:
        dev = torch.device("cuda", device_index)
        a = torch.randn(16, 16, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(16, 16, device=dev).to(torch.float8_e4m3fn).t()
        one = torch.ones((), device=dev)
        torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
        return True, ""
    except Exception as e:                                     # unsupported arch / torch build
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def fp8_available(device: torch.device) -> tuple[bool, str]:
    """(usable, reason): can ``torch._scaled_mm`` run on ``device``? Functional probe, cached per CUDA device."""
    device = torch.device(device)
    if device.type != "cuda":
        return False, f"fp8 GEMMs need a CUDA device, got {device}"
    return _probe(device.index if device.index is not None else torch.cuda.current_device())


def _to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype):
    """Tensorwise dynamic quantisation: (fp8 data, inverse scale) with amax mapped to the fp8 max."""
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().max()
    scale = (fp8_max / amax.double().clamp(min=_EPS)).float()  # float64 division: compile/eager-consistent
    x_fp8 = (x.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale.reciprocal()


def _col_major(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous().t()


def _pad_rows(x: torch.Tensor, mult: int = 16) -> torch.Tensor:
    """Zero-pad dim 0 to a multiple of `mult` (a _scaled_mm requirement on the inner dim of the grad_weight GEMM;
    zero rows contribute nothing to that sum)."""
    pad = (-x.shape[0]) % mult
    return F.pad(x, (0, 0, 0, pad)) if pad else x


@torch._dynamo.allow_in_graph
class _Fp8LinearFn(torch.autograd.Function):
    """y = x @ W.T with both operands fp8 (e4m3) and an fp32 accumulator; fp8 (e5m2-gradient) backward."""

    @staticmethod
    def forward(ctx, x2: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype):
        x_fp8, x_inv = _to_fp8(x2, torch.float8_e4m3fn)
        w_fp8, w_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(x_fp8, x_inv, w_fp8, w_inv)
        ctx.x_dtype = x2.dtype
        return torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_inv, scale_b=w_inv, out_dtype=out_dtype,
                                use_fast_accum=False)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_fp8, x_inv, w_fp8, w_inv = ctx.saved_tensors
        g_fp8, g_inv = _to_fp8(grad_out, torch.float8_e5m2)
        # grad_input = grad_out @ W            [M, N] @ [N, K] -> [M, K]
        grad_x = torch._scaled_mm(g_fp8, _col_major(w_fp8), scale_a=g_inv, scale_b=w_inv, out_dtype=ctx.x_dtype,
                                  use_fast_accum=False)
        # grad_weight = grad_out.T @ x         [N, M] @ [M, K] -> [N, K], fp32 for the fp32 master
        g_t = _pad_rows(g_fp8.view(torch.uint8)).view(torch.float8_e5m2).t().contiguous()
        x_p = _col_major(_pad_rows(x_fp8.view(torch.uint8)).view(torch.float8_e4m3fn))
        grad_w = torch._scaled_mm(g_t, x_p, scale_a=g_inv, scale_b=x_inv, out_dtype=torch.float32,
                                  use_fast_accum=False)
        return grad_x, grad_w, None


def fp8_linear(x: torch.Tensor, linear: torch.nn.Linear, out_dtype: torch.dtype,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """``linear(x)`` with the GEMM in fp8 (see module docstring). ``weight`` overrides ``linear.weight`` (e.g. with a
    per-output-channel scale folded in); the bias, if any, is added afterwards in ``out_dtype``."""
    w = linear.weight if weight is None else weight
    lead = x.shape[:-1]
    y = _Fp8LinearFn.apply(x.reshape(-1, x.shape[-1]), w, out_dtype)
    y = y.reshape(*lead, y.shape[-1])
    if linear.bias is not None:
        y = y + linear.bias.to(y.dtype)
    return y
