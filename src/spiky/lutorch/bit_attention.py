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


def _native_has_bit_attn_tc() -> bool:
    native = _get_native()
    return native is not None and hasattr(native, "bit_attn_flash_forward_tc")


# Forward path selection via env var:
#   unset/"0": bit-packed flash CUDA kernel (no Tensor Cores) + F.SDPA
#              fallback. Exact fp32; tests pass at atol=1e-5.
#   "tc"/"1":  CUTLASS-tuned bit-packed flash kernel — uses Hopper's binary
#              Tensor Cores (m8n8k128 b1 XOR-popcount) for Q@K^T and bf16
#              Tensor Cores (m8n32k16) for A@V. ~2.5x faster than the
#              non-TC kernel; beats cuDNN at BH=24 d=496. ~2e-3 relative
#              precision (bf16 quantization on A and V).
_FORWARD_KERNEL_MODE = os.environ.get("SPIKY_BIT_ATTN_USE_FORWARD_KERNEL", "0")


def _bit_attn_flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale,                           # float OR 0-d tensor (TC kernel reads tensor on-device)
    is_causal: bool,
) -> torch.Tensor:
    """Forward path dispatcher.

    Default: bit-packed flash CUDA kernel (no TC). Set
    SPIKY_BIT_ATTN_USE_FORWARD_KERNEL=tc for the binary-TC + bf16-WMMA path.
    """
    native = _get_native()
    if (
        _FORWARD_KERNEL_MODE in ("tc", "1")
        and native is not None
        and hasattr(native, "bit_attn_flash_forward_tc")
        and q.is_cuda and k.is_cuda and v.is_cuda
        and q.dtype == torch.float32
        and k.dtype == torch.float32
        and v.dtype == torch.float32
    ):
        orig_shape_prefix = q.shape[:-2]
        T = q.shape[-2]
        d = q.shape[-1]
        d_v = v.shape[-1]
        BH = int(torch.tensor(orig_shape_prefix).prod().item()) if len(orig_shape_prefix) else 1
        if d_v <= 128:
            q3 = q.reshape(BH, T, d).contiguous()
            k3 = k.reshape(BH, T, d).contiguous()
            v3 = v.reshape(BH, T, d_v).contiguous()
            # TC kernel reads scale on-device (no .item() sync). Convert
            # caller's scale to a 0-d fp32 CUDA tensor.
            if isinstance(scale, torch.Tensor):
                scale_t = scale.detach().to(device=q.device, dtype=torch.float32).contiguous().view(())
            else:
                scale_t = torch.tensor(float(scale), device=q.device, dtype=torch.float32)
            o3 = native.bit_attn_flash_forward_tc(q3, k3, v3, scale_t, bool(is_causal))
            return o3.reshape(*orig_shape_prefix, T, d_v)
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
    compute_grad_scale: bool = False,
):
    """Explicit SDPA backward (PyTorch reference).

    When `compute_grad_scale=True`, also returns grad_scale (for learnable
    `scale` parameter): grad_scale = sum(dS * (Q @ K^T)) — derived from
    ∂S/∂scale = Q @ K^T since S = Q @ K^T * scale.
    """
    T = q.shape[-2]
    if use_bf16_matmul:
        q_bf = q.to(torch.bfloat16)
        k_bf = k.to(torch.bfloat16)
        s_unscaled = torch.matmul(q_bf, k_bf.transpose(-2, -1)).to(torch.float32)
    else:
        s_unscaled = torch.matmul(q, k.transpose(-2, -1))
    s = s_unscaled * scale
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=s.device), diagonal=1,
        )
        s = s.masked_fill(causal_mask, float('-inf'))
    a = torch.softmax(s, dim=-1)
    a = torch.nan_to_num(a, nan=0.0)

    dv = torch.matmul(a.transpose(-2, -1), grad_o)
    da = torch.matmul(grad_o, v.transpose(-2, -1))
    ds = a * (da - (a * da).sum(dim=-1, keepdim=True))
    if use_bf16_matmul:
        ds_bf = ds.to(torch.bfloat16)
        dq = torch.matmul(ds_bf, k_bf).to(torch.float32) * scale
        dk = torch.matmul(ds_bf.transpose(-2, -1), q_bf).to(torch.float32) * scale
    else:
        dq = torch.matmul(ds, k) * scale
        dk = torch.matmul(ds.transpose(-2, -1), q) * scale

    if compute_grad_scale:
        # Masked positions have A=0 (softmax of -inf), so dS=0 there → no
        # contribution. Just sum over all elements.
        grad_scale = (ds * s_unscaled).sum()
        return dq, dk, dv, grad_scale
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
    compute_grad_scale: bool = False,
):
    """Backward dispatcher (see `_BACKWARD_KERNEL_MODE` for the modes)."""
    mode = _BACKWARD_KERNEL_MODE
    if mode == "0" or mode == "":
        return _bit_attn_backward_explicit(
            q, k, v, grad_o, scale, is_causal,
            compute_grad_scale=compute_grad_scale,
        )
    if mode == "bf16" or mode == "1":
        return _get_compiled_bf16_backward()(
            q, k, v, grad_o, scale, is_causal,
            use_bf16_matmul=True,
            compute_grad_scale=compute_grad_scale,
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
        # `scale` may be a 0-d tensor (learnable scalar Parameter) or a
        # Python float. The TC forward path passes the tensor through to
        # CUDA (no host-device sync); only fp32 fallback path needs a
        # Python value, in which case we accept the .item() cost there.
        ctx.scale_was_tensor = isinstance(scale, torch.Tensor)
        ctx.save_for_backward(q, k, v)
        ctx.scale_obj = scale  # tensor or float; backward extracts value as needed
        ctx.is_causal = bool(is_causal)
        return _bit_attn_flash_forward(q, k, v, scale, bool(is_causal))

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v = ctx.saved_tensors
        needs_q, needs_k, needs_v, needs_scale, _ = ctx.needs_input_grad
        if not (needs_q or needs_k or needs_v or needs_scale):
            return None, None, None, None, None
        # Pass scale through as-is (tensor or float) — PyTorch ops handle
        # broadcasting from a 0-d tensor, so no .item() sync is needed.
        scale_obj = ctx.scale_obj
        compute_scale = ctx.scale_was_tensor and needs_scale
        if compute_scale:
            dq, dk, dv, dscale = _bit_attn_backward_dispatch(
                q, k, v, grad_o.contiguous(), scale_obj, ctx.is_causal,
                compute_grad_scale=True,
            )
        else:
            dq, dk, dv = _bit_attn_backward_dispatch(
                q, k, v, grad_o.contiguous(), scale_obj, ctx.is_causal,
            )
            dscale = None
        return (
            dq if needs_q else None,
            dk if needs_k else None,
            dv if needs_v else None,
            dscale if compute_scale else None,
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
        scale=None,
    ) -> torch.Tensor:
        """If `scale` is provided (Python float OR 0-d tensor), it overrides
        the module's init-time scale. When passed as a tensor with
        requires_grad=True (e.g. a learnable Parameter), grad flows back
        to it via `grad_scale = sum(dS * (Q @ K^T))`."""
        s = scale if scale is not None else self.scale
        return _BitAttentionFn.apply(q, k, v, s, is_causal)
