"""BitAttention — scaled dot-product attention for ±1 queries and keys.

Semantic: mathematically identical to F.scaled_dot_product_attention when
q, k are exact ±1. The win is the fused flash-attention-style CUDA forward
kernel that packs q, k to bits (+1→0, −1→1) and replaces the Q·K^T matmul
with popcount(Q_bits XOR K_bits):

    q · k = d − 2·popcount(q XOR k)         for q, k ∈ {−1, +1}^d

- Forward: CUDA kernel on GPU when available, `F.scaled_dot_product_attention`
  fallback on CPU / when the native extension is missing.
- Backward: explicit closed-form gradients (dV, dA, dS, dQ, dK). The matmuls
  `dQ = dS @ K * scale` and `dK = dS^T @ Q * scale` have a ±1 operand (K, Q)
  so they are signed accumulations in principle — see
  `bit_attn_flash_bwd_kernel` (TODO) for the CUDA realization. The PyTorch
  reference here uses standard matmul; numerical output is identical to
  F.scaled_dot_product_attention's autograd, but the call structure makes
  the bit-trick boundary explicit.
"""
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_native():
    try:
        from lutorch_cuda import get_lutorch_manager
        return get_lutorch_manager()
    except Exception:
        return None


def _native_has_bit_attn() -> bool:
    native = _get_native()
    return native is not None and hasattr(native, "bit_attn_flash_forward")


def _native_has_bit_attn_backward() -> bool:
    native = _get_native()
    return native is not None and hasattr(native, "bit_attn_backward")


def _bit_attn_flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Forward path dispatcher.

    Uses the bit-packed flash CUDA kernel when available (universal in T —
    flash-style streaming, no T² intermediates); falls back to
    F.scaled_dot_product_attention otherwise.
    """
    native = _get_native()
    use_kernel = (
        q.is_cuda and k.is_cuda and v.is_cuda
        and q.dtype == torch.float32
        and k.dtype == torch.float32
        and v.dtype == torch.float32
        and native is not None
        and hasattr(native, "bit_attn_flash_forward")
    )
    if not use_kernel:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scale,
        )

    orig_shape_prefix = q.shape[:-2]
    T = q.shape[-2]
    d = q.shape[-1]
    d_v = v.shape[-1]
    BH = int(torch.tensor(orig_shape_prefix).prod().item()) if len(orig_shape_prefix) else 1

    q3 = q.reshape(BH, T, d).contiguous()
    k3 = k.reshape(BH, T, d).contiguous()
    v3 = v.reshape(BH, T, d_v).contiguous()

    # Kernel's supported size envelope: n_words <= 16, d_v <= 128.
    n_words = (d + 31) // 32
    if n_words > 16 or d_v > 128:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scale,
        )

    o3 = native.bit_attn_flash_forward(q3, k3, v3, float(scale), bool(is_causal))
    return o3.reshape(*orig_shape_prefix, T, d_v)


def _bit_attn_backward_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_o: torch.Tensor,
    scale: float,
    is_causal: bool,
    use_bf16_matmul: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit SDPA backward (PyTorch reference).

    Forward (recap):
        S = Q @ K^T * scale            (with -inf at j > i if causal)
        A = softmax(S, dim=-1)
        O = A @ V

    Backward (given dO):
        dV = A^T @ dO
        dA = dO @ V^T
        dS = A * (dA - rowsum(A * dA))               # softmax backward
        dQ = (dS @ K) * scale
        dK = (dS^T @ Q) * scale

    The matmuls `dQ = dS @ K * scale` and `dK = dS^T @ Q * scale` have one
    ±1 operand (K, Q respectively). When `use_bf16_matmul=True`, dQ/dK are
    computed via bf16 cuBLAS GEMM — on Hopper this dispatches to WGMMA,
    yielding ~2x speedup vs fp32 sgemm at the cost of ~1e-3 relative
    precision (K, Q are ±1 so their bf16 cast is lossless; dS picks up
    the bf16 quantization).
    """
    T = q.shape[-2]
    if use_bf16_matmul:
        # S = Q @ K^T * scale uses bf16 cuBLAS GEMM (WGMMA on Hopper). Q, K
        # are ±1 → bf16 cast is lossless, so this is a pure throughput win.
        q_bf = q.to(torch.bfloat16)
        k_bf = k.to(torch.bfloat16)
        s = torch.matmul(q_bf, k_bf.transpose(-2, -1)).to(torch.float32) * scale
    else:
        s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=s.device), diagonal=1,
        )
        s = s.masked_fill(causal_mask, float('-inf'))
    a = torch.softmax(s, dim=-1)
    # Fully-masked rows are impossible under is_causal (i can always attend
    # to itself), but guard against NaN propagation if they appear.
    a = torch.nan_to_num(a, nan=0.0)

    dv = torch.matmul(a.transpose(-2, -1), grad_o)
    da = torch.matmul(grad_o, v.transpose(-2, -1))
    ds = a * (da - (a * da).sum(dim=-1, keepdim=True))
    if use_bf16_matmul:
        ds_bf = ds.to(torch.bfloat16)
        # Skip explicit .contiguous() on dS^T — cuBLAS handles strided inputs.
        dq = torch.matmul(ds_bf, k_bf).to(torch.float32) * scale
        dk = torch.matmul(ds_bf.transpose(-2, -1), q_bf).to(torch.float32) * scale
    else:
        dq = torch.matmul(ds, k) * scale
        dk = torch.matmul(ds.transpose(-2, -1), q) * scale
    return dq, dk, dv


# Backward path selection via env var:
#   unset/"0": fp32 PyTorch reference (exact, matches SDPA autograd)
#   "bf16":    fp32 components except S/dQ/dK via bf16 cuBLAS GEMM. On
#              Hopper this dispatches to WGMMA → ~2.4x speedup vs fp32
#              cuBLAS sgemm. Eltwise ops fused via torch.compile for an
#              additional ~1.4-1.8x → ~3-4x total speedup vs fp32. K, Q
#              are ±1 (lossless bf16 cast); dS picks up ~1e-3 relative
#              quantization error.
#   "wmma":    custom WMMA kernel with bit-packed K/Q (slower than bf16
#              cuBLAS on H100, but exercises the ±1-storage code path —
#              useful for deployment hardware without Tensor Cores).
_BACKWARD_KERNEL_MODE = os.environ.get("SPIKY_BIT_ATTN_USE_BACKWARD_KERNEL", "0")

# Lazy-compiled bf16 backward (torch.compile fuses eltwise ops + scheduling).
_compiled_bf16_backward: Optional[callable] = None

def _get_compiled_bf16_backward():
    global _compiled_bf16_backward
    if _compiled_bf16_backward is None:
        _compiled_bf16_backward = torch.compile(
            _bit_attn_backward_explicit, dynamic=True,
        )
    return _compiled_bf16_backward


def _bit_attn_backward_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_o: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward dispatcher (see `_BACKWARD_KERNEL_MODE` for the modes)."""
    mode = _BACKWARD_KERNEL_MODE
    if mode == "0" or mode == "":
        return _bit_attn_backward_explicit(q, k, v, grad_o, scale, is_causal)
    if mode == "bf16" or mode == "1":
        return _get_compiled_bf16_backward()(
            q, k, v, grad_o, scale, is_causal, use_bf16_matmul=True,
        )

    # mode == "wmma": dispatch to native CUDA WMMA kernel
    native = _get_native()
    use_kernel = (
        q.is_cuda and k.is_cuda and v.is_cuda and grad_o.is_cuda
        and q.dtype == torch.float32
        and k.dtype == torch.float32
        and v.dtype == torch.float32
        and grad_o.dtype == torch.float32
        and native is not None
        and hasattr(native, "bit_attn_backward")
    )
    if not use_kernel:
        return _bit_attn_backward_explicit(q, k, v, grad_o, scale, is_causal)

    # Kernel requires 3-D tensors [BH, T, feat]. Collapse leading dims.
    orig_shape_prefix = q.shape[:-2]
    T = q.shape[-2]
    d = q.shape[-1]
    d_v = v.shape[-1]
    BH = int(torch.tensor(orig_shape_prefix).prod().item()) if len(orig_shape_prefix) else 1

    # Backward kernel has no n_words limit (unlike forward): kernels read
    # one word per t per d-tile, no per-thread register arrays.
    q3 = q.reshape(BH, T, d).contiguous()
    k3 = k.reshape(BH, T, d).contiguous()
    v3 = v.reshape(BH, T, d_v).contiguous()
    go3 = grad_o.reshape(BH, T, d_v).contiguous()

    dq3, dk3, dv3 = native.bit_attn_backward(q3, k3, v3, go3, float(scale), bool(is_causal))
    return (
        dq3.reshape(*orig_shape_prefix, T, d),
        dk3.reshape(*orig_shape_prefix, T, d),
        dv3.reshape(*orig_shape_prefix, T, d_v),
    )


class _BitAttentionFn(torch.autograd.Function):
    """Custom autograd boundary.

    Forward: bit-packed CUDA kernel (exact for ±1 q, k) or SDPA fallback.
    Backward: explicit closed-form gradients (no autograd-through-SDPA
    re-run). The PyTorch reference uses standard matmul; the
    `dQ = dS @ K * scale` and `dK = dS^T @ Q * scale` matmuls have a ±1
    operand and are the slot for a future signed-accumulation CUDA kernel
    (`bit_attn_flash_bwd_kernel`). Numerically matches SDPA's autograd to
    fp32 rollup tolerance.
    """

    @staticmethod
    def forward(ctx, q, k, v, scale, is_causal):
        ctx.save_for_backward(q, k, v)
        ctx.scale = float(scale)
        ctx.is_causal = bool(is_causal)
        return _bit_attn_flash_forward(q, k, v, float(scale), bool(is_causal))

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v = ctx.saved_tensors
        needs_q, needs_k, needs_v = ctx.needs_input_grad[:3]
        if not (needs_q or needs_k or needs_v):
            return None, None, None, None, None
        dq, dk, dv = _bit_attn_backward_dispatch(
            q, k, v, grad_o.contiguous(), ctx.scale, ctx.is_causal,
        )
        return (
            dq if needs_q else None,
            dk if needs_k else None,
            dv if needs_v else None,
            None,
            None,
        )


class BitAttention(nn.Module):
    """Scaled dot-product attention specialised for ±1 queries/keys.

    Args:
        d: last-dim size of q/k (used to default `scale = 1/√d`).
        scale: explicit softmax pre-scale; defaults to `1/√d` when None.

    Forward:
        q, k: (..., T, d)    float; expected ±1. No runtime check.
        v:    (..., T, d_v)  float; arbitrary values.
        is_causal: apply lower-triangular mask.

    Returns: (..., T, d_v) float.

    Kernel envelope (current): d ≤ 512, d_v ≤ 128. Outside this range, falls
    back to F.scaled_dot_product_attention transparently.
    """

    def __init__(self, d: int, scale: Optional[float] = None):
        super().__init__()
        self.d = int(d)
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(d)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        return _BitAttentionFn.apply(q, k, v, self.scale, is_causal)
