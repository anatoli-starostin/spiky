"""BitAttention — scaled dot-product attention for ±1 queries and keys.

Semantic: mathematically identical to F.scaled_dot_product_attention when
q, k are exact ±1. The win is the fused flash-attention-style CUDA forward
kernel that packs q, k to bits (+1→0, −1→1) and replaces the Q·K^T matmul
with popcount(Q_bits XOR K_bits):

    q · k = d − 2·popcount(q XOR k)         for q, k ∈ {−1, +1}^d

- Forward: CUDA kernel on GPU when available, `F.scaled_dot_product_attention`
  fallback on CPU / when the native extension is missing.
- Backward: re-runs SDPA on the saved float q, k, v for autograd gradients.
  Since the forward is mathematically identical to SDPA, the gradients match
  exactly.
"""
import math
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


def _bit_attn_flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Forward path dispatcher. Uses the CUDA kernel if available; otherwise
    falls back to F.scaled_dot_product_attention (mathematically identical).

    Accepts q, k, v of shape (..., T, feat). The kernel requires 3-D input
    [BH, T, feat], so we collapse leading dims before dispatching and
    restore them on return.
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


class _BitAttentionFn(torch.autograd.Function):
    """Custom autograd boundary.

    Forward: bit-packed CUDA kernel (exact for ±1 q, k) or SDPA fallback.
    Backward: re-runs F.scaled_dot_product_attention with requires_grad on
    the saved q, k, v and returns the standard gradients. This works because
    the kernel's forward is mathematically identical to SDPA — the gradients
    of SDPA are the correct gradients for our forward.
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
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(needs_q)
            k_ = k.detach().requires_grad_(needs_k)
            v_ = v.detach().requires_grad_(needs_v)
            out = F.scaled_dot_product_attention(
                q_, k_, v_, is_causal=ctx.is_causal, scale=ctx.scale,
            )
            want = [t for t, need in zip((q_, k_, v_), (needs_q, needs_k, needs_v)) if need]
            grads = torch.autograd.grad(out, want, grad_outputs=grad_o)
        it = iter(grads)
        dq = next(it) if needs_q else None
        dk = next(it) if needs_k else None
        dv = next(it) if needs_v else None
        return dq, dk, dv, None, None


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
