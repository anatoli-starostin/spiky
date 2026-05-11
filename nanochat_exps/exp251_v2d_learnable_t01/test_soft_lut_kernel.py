"""Test new `soft_lut_backward_grad_x` CUDA kernel against pure-Python soft path.

Pipeline under test:
  1) compute index from sign bits, embedding_bag forward (TinyMHLut-style)
  2) backward:
     - dL/dweights via existing `tiny_mhlut_backward_na1` (sparse scatter)
     - dL/dsel_soft via torch.einsum (cuBLAS)
     - dL/dx, dL/dlog_T_soft, dL/dlog_T_sel via NEW `soft_lut_backward_grad_x` kernel

Equivalence target: SoftMultiHeadLUT(hard=True) reference (same bit_matrix
convention).
"""
import math
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

import lutorch_cuda


def _bit_matrix_msb(nap, device, dtype=torch.float32):
    n = 1 << nap
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


def _msb_powers(nap, device):
    return (1 << torch.arange(nap - 1, -1, -1, device=device, dtype=torch.int64))


class _SoftLUTKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, anchor_pairs_a, anchor_pairs_b,
                log_T_soft, log_T_sel, n_heads, tables_per_head, use_bf16):
        B, input_dim = x.shape
        n_tables, nap = anchor_pairs_a.shape
        table_dim = 1 << nap
        n_outputs = weights.shape[2]

        idx_a = anchor_pairs_a.long(); idx_b = anchor_pairs_b.long()
        x_a = x[:, idx_a]; x_b = x[:, idx_b]
        d = x_a - x_b                                                       # [B, T, NAP]
        bits = (d > 0).to(torch.int64)
        powers = _msb_powers(nap, x.device).view(1, 1, -1)
        index = (bits * powers).sum(dim=-1)                                 # [B, T] int64

        # Forward via embedding_bag
        weights_flat = weights.view(n_tables * table_dim, n_outputs)
        table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
        flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
        n_bags = B * n_heads
        offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tables_per_head
        out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
        out = out_flat.view(B, n_heads, n_outputs)

        # Save for backward — note we save d in the dtype we'll feed into the kernel.
        if use_bf16 and x.is_cuda:
            d_save = d.to(torch.bfloat16)
        else:
            d_save = d.contiguous()
        ctx.save_for_backward(x, d_save, weights, index, anchor_pairs_a, anchor_pairs_b,
                              log_T_soft, log_T_sel)
        ctx.n_heads = n_heads
        ctx.tables_per_head = tables_per_head
        ctx.use_bf16 = use_bf16
        ctx.input_dim = input_dim
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, d_save, weights, index, anchor_pairs_a, anchor_pairs_b,
         log_T_soft, log_T_sel) = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tables_per_head
        B = x.shape[0]
        n_tables, nap = anchor_pairs_a.shape
        n_outputs = weights.shape[2]

        T_soft = float(log_T_soft.detach().exp().item())
        T_sel  = float(log_T_sel.detach().exp().item())

        # 1) dL/dweights via native tiny_mhlut_backward_na1 (sparse scatter).
        #    `lookup_alt_indices` arg is unused for our purposes — pass `index`
        #    and ignore the returned carrier grads.
        mgr = lutorch_cuda.get_lutorch_manager()
        grad_weights, _gm, _ga = mgr.tiny_mhlut_backward_na1(
            grad_out.contiguous().to(weights.dtype),
            weights, index.contiguous(), index.contiguous(), tph,
        )

        # 2) dL/dsel_soft via cuBLAS — broadcast grad_out over tph and matmul.
        #    Run under bf16 autocast to match SoftMHLut's compute precision.
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if ctx.use_bf16 and x.is_cuda else contextlib.nullcontext()
        )
        with autocast_ctx:
            grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
            d_sel_soft = torch.einsum("bto,tko->btk", grad_pt, weights)     # [B, T, K]

        # 3) dL/dx, dL/dlog_T_soft, dL/dlog_T_sel via the new kernel.
        d_buf = d_save  # already in correct dtype (bf16 if use_bf16)
        # Make d_buf and d_sel_soft same dtype if needed
        if d_buf.dtype != d_sel_soft.dtype:
            d_buf = d_buf.to(d_sel_soft.dtype)

        grad_x_fp32, d_log_T_soft, d_log_T_sel = mgr.soft_lut_backward_grad_x(
            d_sel_soft.contiguous(), d_buf.contiguous(),
            anchor_pairs_a, anchor_pairs_b,
            ctx.input_dim, T_soft, T_sel,
        )
        grad_x = grad_x_fp32.to(x.dtype)

        return (grad_x, grad_weights, None, None,
                d_log_T_soft, d_log_T_sel, None, None, None)


# --- Reference: SoftMultiHeadLUT(hard=True) using same bit_matrix convention ---
def soft_reference_forward(x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                           T_soft, T_sel, n_heads, tables_per_head, use_bf16=False):
    B = x.shape[0]
    n_tables, nap = anchor_pairs_a.shape
    n_outputs = weights.shape[2]
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if use_bf16 and x.is_cuda else contextlib.nullcontext())
    with autocast_ctx:
        idx_a = anchor_pairs_a.long(); idx_b = anchor_pairs_b.long()
        x_a = x[:, idx_a]; x_b = x[:, idx_b]
        rd = x_a - x_b
        p = rd / (T_soft + rd.abs())
        ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
        sel_soft = F.softmax(ts / T_sel, dim=-1)
        idx = sel_soft.argmax(dim=-1, keepdim=True)
        sel_hard = torch.zeros_like(sel_soft).scatter_(-1, idx, 1.0)
        sel = sel_hard - sel_soft.detach() + sel_soft
        out_t = torch.einsum("btk,tko->bto", sel, weights)
    out_t = out_t.to(weights.dtype)
    return out_t.view(B, n_heads, tables_per_head, n_outputs).sum(dim=2)


def make(input_dim, n_heads, tph, nap, n_outputs, device, weight_dtype=torch.float32):
    n_tables = n_heads * tph
    rng = torch.Generator(device=device).manual_seed(0)
    a = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b = torch.where(b == a, (b + 1) % input_dim, b)
    weights = ((torch.rand(n_tables, 1 << nap, n_outputs, generator=rng, device=device) - 0.5) * 0.002)
    weights = weights.to(weight_dtype).clone().requires_grad_(True)
    bm = _bit_matrix_msb(nap, device, dtype=weight_dtype)
    return weights, a.to(torch.int16), b.to(torch.int16), bm


def equiv_check(device):
    # Use a small case for fp32 comparison (no autocast).
    torch.manual_seed(0)
    B, input_dim, n_heads, tph, nap, n_outputs = 16, 64, 4, 8, 6, 12
    weights, ap_a, ap_b, bm = make(input_dim, n_heads, tph, nap, n_outputs, device)
    log_T_soft = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel  = torch.tensor(math.log(0.5), device=device, requires_grad=True)

    x_a = torch.randn(B, input_dim, device=device, requires_grad=True)
    out_ref = soft_reference_forward(x_a, weights, ap_a, ap_b, bm,
                                     log_T_soft.exp(), log_T_sel.exp(),
                                     n_heads, tph, use_bf16=False)
    g_ref = torch.autograd.grad(out_ref.sum(), [x_a, weights, log_T_soft, log_T_sel])

    weights2 = weights.detach().clone().requires_grad_(True)
    log_T_soft2 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel2  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    out_new = _SoftLUTKernel.apply(x_b, weights2, ap_a, ap_b,
                                   log_T_soft2, log_T_sel2,
                                   n_heads, tph, False)  # use_bf16=False for fp32 comparison
    g_new = torch.autograd.grad(out_new.sum(), [x_b, weights2, log_T_soft2, log_T_sel2])

    print(f"\n=== Equivalence (fp32, NAP={nap}) ===")
    print(f"  out abs|Δ|max  = {(out_ref - out_new).abs().max().item():.2e}")
    for name, a, b in zip(["g_x", "g_w", "g_logTs", "g_logTx"], g_ref, g_new):
        diff = (a - b).abs().max().item()
        ref = a.abs().max().item() if a.dim() else abs(a.item())
        rel = diff / max(ref, 1e-12)
        print(f"  {name:8s}  abs|Δ|max = {diff:.2e}   ref|max| = {ref:.2e}   rel = {rel:.2e}")

    # Bigger NAP fp32 (no autocast)
    print(f"\n=== Equivalence (fp32, NAP=8, B=128) ===")
    torch.manual_seed(0)
    weights_fp, ap_a_fp, ap_b_fp, bm_fp = make(96, 6, 256, 8, 32, device)
    log_T_soft_fp = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel_fp  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x_fp = torch.randn(128, 96, device=device, requires_grad=True)
    out_ref_fp = soft_reference_forward(x_fp, weights_fp, ap_a_fp, ap_b_fp, bm_fp,
                                        log_T_soft_fp.exp(), log_T_sel_fp.exp(),
                                        6, 256, use_bf16=False)
    g_ref_fp = torch.autograd.grad(out_ref_fp.sum(), [x_fp, weights_fp, log_T_soft_fp, log_T_sel_fp])
    weights_fp2 = weights_fp.detach().clone().requires_grad_(True)
    log_T_soft_fp2 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel_fp2  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x_fp2 = x_fp.detach().clone().requires_grad_(True)
    out_new_fp = _SoftLUTKernel.apply(x_fp2, weights_fp2, ap_a_fp, ap_b_fp,
                                      log_T_soft_fp2, log_T_sel_fp2, 6, 256, False)
    g_new_fp = torch.autograd.grad(out_new_fp.sum(), [x_fp2, weights_fp2, log_T_soft_fp2, log_T_sel_fp2])
    print(f"  out abs|Δ|max  = {(out_ref_fp - out_new_fp).abs().max().item():.2e}")
    for name, a, b in zip(["g_x", "g_w", "g_logTs", "g_logTx"], g_ref_fp, g_new_fp):
        diff = (a - b).abs().max().item()
        ref = a.abs().max().item() if a.dim() else abs(a.item())
        rel = diff / max(ref, 1e-12)
        print(f"  {name:8s}  abs|Δ|max = {diff:.2e}   ref|max| = {ref:.2e}   rel = {rel:.2e}")

    # Bigger NAP test (matches v_lut shape NAP=8) under bf16 autocast.
    print(f"\n=== Equivalence (bf16 autocast, NAP=8, B=128) ===")
    torch.manual_seed(0)
    weights3, ap_a3, ap_b3, bm3 = make(96, 6, 256, 8, 32, device)
    log_T_soft3 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel3  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x3 = torch.randn(128, 96, device=device, requires_grad=True)

    out_ref3 = soft_reference_forward(x3, weights3, ap_a3, ap_b3, bm3,
                                      log_T_soft3.exp(), log_T_sel3.exp(),
                                      6, 256, use_bf16=True)
    g_ref3 = torch.autograd.grad(out_ref3.sum(), [x3, weights3, log_T_soft3, log_T_sel3])

    weights4 = weights3.detach().clone().requires_grad_(True)
    log_T_soft4 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel4  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x4 = x3.detach().clone().requires_grad_(True)
    out_new4 = _SoftLUTKernel.apply(x4, weights4, ap_a3, ap_b3,
                                    log_T_soft4, log_T_sel4, 6, 256, True)
    g_new4 = torch.autograd.grad(out_new4.sum(), [x4, weights4, log_T_soft4, log_T_sel4])

    print(f"  out abs|Δ|max  = {(out_ref3 - out_new4).abs().max().item():.2e}")
    for name, a, b in zip(["g_x", "g_w", "g_logTs", "g_logTx"], g_ref3, g_new4):
        diff = (a - b).abs().max().item()
        ref = a.abs().max().item() if a.dim() else abs(a.item())
        rel = diff / max(ref, 1e-12)
        print(f"  {name:8s}  abs|Δ|max = {diff:.2e}   ref|max| = {ref:.2e}   rel = {rel:.2e}")


if __name__ == "__main__":
    dev = torch.device("cuda")
    equiv_check(dev)
