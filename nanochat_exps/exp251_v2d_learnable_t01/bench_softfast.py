"""Prototype + correctness test + bench of TinyMHLut-fast forward + soft backward.

Custom autograd Function:
  Forward:   index = bit-pack of sign(x_a - x_b), then embedding_bag(weights, index)
             [identical output to SoftMultiHeadLUT(hard=True), since
              argmax(softmax(ts/T_sel)) = argmax(ts) = sign-bit packed index]
  Backward:  recompute p, ts, sel_soft on demand; propagate gradients matching
             the original SoftMultiHeadLUT backward (sparse weights via scatter,
             smooth dL/dx via rational sign + softmax).

Reference: a re-implementation of SoftMHLut(hard=True) using the SAME bit
convention so we can compare forward + gradients element-wise.
"""
import math
import time
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bit_matrix_msb(nap, device, dtype=torch.float32):
    """Same convention as soft_multi_head_lut._bit_matrix: bit i of bit_matrix
    corresponds to bit (nap-1-i) of integer k. MSB-first."""
    n = 1 << nap
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


def _msb_powers(nap, device):
    """powers[i] = 2^(nap-1-i): index = sum_i (d_i > 0) * 2^(nap-1-i)."""
    return (1 << torch.arange(nap - 1, -1, -1, device=device, dtype=torch.int64))


def _fwd_body(x, weights, anchor_pairs_a, anchor_pairs_b, n_heads, tables_per_head):
    B, _ = x.shape
    n_tables, nap = anchor_pairs_a.shape
    table_dim = weights.shape[1]
    n_outputs = weights.shape[2]
    idx_a = anchor_pairs_a.long(); idx_b = anchor_pairs_b.long()
    x_a = x[:, idx_a]; x_b = x[:, idx_b]
    d = x_a - x_b
    bits = (d > 0).to(torch.int64)
    powers = _msb_powers(nap, x.device).view(1, 1, -1)
    index = (bits * powers).sum(dim=-1)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tables_per_head
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs), d, index


def _bwd_body(grad_out, d, weights, index, anchor_pairs_a, anchor_pairs_b, bit_matrix,
              log_T_soft, log_T_sel, n_heads, tables_per_head, use_bf16, x_dtype):
    B = d.shape[0]
    n_tables, nap = anchor_pairs_a.shape
    table_dim = bit_matrix.shape[1]
    n_outputs = weights.shape[2]
    input_dim = grad_out.new_zeros(0).numel() + d.shape[0]  # placeholder — real handled by caller
    T_soft = log_T_soft.exp()
    T_sel = log_T_sel.exp()

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 and d.is_cuda
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        denom = T_soft + d.abs()
        p = d / denom
        ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
        z = ts / T_sel
        sel_soft = F.softmax(z, dim=-1)
        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tables_per_head, n_outputs).reshape(B, n_tables, n_outputs)
        d_sel_soft = torch.einsum("bto,tko->btk", grad_pt, weights)
        sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
        d_z = sel_soft * (d_sel_soft - sum_term)
        d_ts = d_z / T_sel
        d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))
        d_d = d_p * (T_soft / (denom * denom))

    d_d_fp = d_d.to(torch.float32)
    return d_d_fp, d_z, z, grad_pt


_FWD_C = None
_BWD_C = None
def _maybe_compile():
    global _FWD_C, _BWD_C
    if _FWD_C is None:
        _FWD_C = torch.compile(_fwd_body, dynamic=True)
        _BWD_C = torch.compile(_bwd_body, dynamic=True)


class _SoftFastFn(torch.autograd.Function):
    """TinyMHLut-fast forward + SoftMHLut-equivalent soft backward."""

    @staticmethod
    def forward(ctx, x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                log_T_soft, log_T_sel, n_heads, tables_per_head, use_bf16, compiled=False):
        if compiled:
            _maybe_compile()
            out, d, index = _FWD_C(x, weights, anchor_pairs_a, anchor_pairs_b, n_heads, tables_per_head)
        else:
            out, d, index = _fwd_body(x, weights, anchor_pairs_a, anchor_pairs_b, n_heads, tables_per_head)
        ctx.save_for_backward(d, weights, index, anchor_pairs_a, anchor_pairs_b,
                              bit_matrix, log_T_soft, log_T_sel)
        ctx.n_heads = n_heads
        ctx.tables_per_head = tables_per_head
        ctx.input_shape = x.shape
        ctx.x_dtype = x.dtype
        ctx.use_bf16 = use_bf16
        ctx.compiled = compiled
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (d, weights, index, anchor_pairs_a, anchor_pairs_b, bit_matrix,
         log_T_soft, log_T_sel) = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tables_per_head
        B, input_dim = ctx.input_shape
        n_tables, nap = anchor_pairs_a.shape
        table_dim = bit_matrix.shape[1]
        n_outputs = weights.shape[2]

        if ctx.compiled:
            d_d_fp, d_z, z, grad_pt = _BWD_C(
                grad_out, d, weights, index, anchor_pairs_a, anchor_pairs_b,
                bit_matrix, log_T_soft, log_T_sel, n_heads, tph, ctx.use_bf16, ctx.x_dtype,
            )
        else:
            d_d_fp, d_z, z, grad_pt = _bwd_body(
                grad_out, d, weights, index, anchor_pairs_a, anchor_pairs_b,
                bit_matrix, log_T_soft, log_T_sel, n_heads, tph, ctx.use_bf16, ctx.x_dtype,
            )
        d_fp = d.to(torch.float32)

        # 1) dL/dweights via sparse scatter at index — fp32 accumulate
        flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
        flat_idx = (index + flat_offset.view(1, -1)).reshape(-1)
        # Convert grad_pt to weights' dtype to match scatter result dtype
        grad_pt_fp = grad_pt.to(weights.dtype)
        grad_w_flat = torch.zeros(n_tables * table_dim, n_outputs,
                                  dtype=weights.dtype, device=weights.device)
        grad_w_flat.index_add_(0, flat_idx, grad_pt_fp.reshape(-1, n_outputs))
        grad_weights = grad_w_flat.view(n_tables, table_dim, n_outputs)

        # 2) dL/dlog_T_sel = -sum(dL/dz * z)  (since T_sel = exp(log_T_sel) and z = ts/T_sel)
        d_log_T_sel = -(d_z.to(torch.float32) * z.to(torch.float32)).sum().reshape(())

        # 3) dL/dlog_T_soft = -sum(d_d * d)  (derived from chain rule, see notes)
        d_log_T_soft = -(d_d_fp * d_fp).sum().reshape(())

        # 4) dL/dx via scatter-add at anchor pairs
        grad_x = torch.zeros(B, input_dim, dtype=ctx.x_dtype, device=d.device)
        idx_a = anchor_pairs_a.long().unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        idx_b = anchor_pairs_b.long().unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        grad_x.scatter_add_(1, idx_a, d_d_fp.reshape(B, -1).to(ctx.x_dtype))
        grad_x.scatter_add_(1, idx_b, -d_d_fp.reshape(B, -1).to(ctx.x_dtype))

        return (grad_x, grad_weights, None, None, None,
                d_log_T_soft, d_log_T_sel, None, None, None, None)


# -------- Narrow custom Function: only `sel @ weights` step --------
class _EmbBagSoftSel(torch.autograd.Function):
    """Replace `einsum("btk,tko->bto", sel, weights)` with embedding_bag forward
    (skips materialising sel and the big matmul) + soft `dL/dsel_soft` backward.

    Forward saves only `weights` (parameter) and `index` (small int64). Compare
    to the original which saves the [B, T, K] sel tensor.
    """
    @staticmethod
    def forward(ctx, sel_soft, weights, n_heads, tph):
        # sel_soft: [B, T, K]   weights: [T, K, O]
        B, T, K = sel_soft.shape
        O = weights.shape[2]
        index = sel_soft.argmax(dim=-1)               # [B, T]
        weights_flat = weights.view(T * K, O)
        table_offset = torch.arange(T, device=weights.device, dtype=index.dtype) * K
        flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
        n_bags = B * n_heads
        offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
        out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
        out = out_flat.view(B, n_heads, O)
        ctx.save_for_backward(weights, index)
        ctx.B = B; ctx.T = T; ctx.K = K; ctx.O = O
        ctx.n_heads = n_heads; ctx.tph = tph
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weights, index = ctx.saved_tensors
        B, T, K, O = ctx.B, ctx.T, ctx.K, ctx.O
        n_heads, tph = ctx.n_heads, ctx.tph
        # Broadcast grad_out across tph tables in each head
        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, O).reshape(B, T, O)
        # dL/dweights via sparse scatter (matches sel_hard.T @ grad math)
        flat_offset = torch.arange(T, device=weights.device, dtype=index.dtype) * K
        flat_idx = (index + flat_offset.view(1, -1)).reshape(-1)
        grad_w_flat = torch.zeros(T * K, O, dtype=weights.dtype, device=weights.device)
        grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, O).to(weights.dtype))
        grad_weights = grad_w_flat.view(T, K, O)
        # dL/dsel_soft via einsum — feeds back through softmax → ts → p → x, T_soft, T_sel
        d_sel_soft = torch.einsum("bto,tko->btk", grad_pt, weights)
        return d_sel_soft, grad_weights, None, None


def soft_then_embbag_forward(x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                             T_soft, T_sel, n_heads, tables_per_head, use_bf16=False):
    """Compile-friendly soft pipeline producing sel_soft, then narrow custom
    Function for the einsum-replacement step. The bulk of the math runs as
    standard PyTorch ops (compilable end-to-end through softmax).
    """
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if use_bf16 and x.is_cuda else contextlib.nullcontext())
    with autocast_ctx:
        idx_a = anchor_pairs_a.long(); idx_b = anchor_pairs_b.long()
        x_a = x[:, idx_a]; x_b = x[:, idx_b]
        d = x_a - x_b
        p = d / (T_soft + d.abs())
        ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
        sel_soft = F.softmax(ts / T_sel, dim=-1)
    out = _EmbBagSoftSel.apply(sel_soft, weights, n_heads, tables_per_head)
    return out.to(weights.dtype)


# -------- Hybrid: embedding_bag forward + phantom soft path for x/T grads --------
def hybrid_forward(x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                   T_soft, T_sel, n_heads, tables_per_head, use_bf16=False):
    """torch-only forward (no custom autograd Function), same gradients as
    reference. Replaces the dominant `einsum(sel, weights)` with
    `embedding_bag(weights, argmax_index)` and grafts a phantom `einsum(sel_soft,
    weights.detach())` whose forward value cancels via STE.
    """
    B, _ = x.shape
    n_tables, nap = anchor_pairs_a.shape
    table_dim = bit_matrix.shape[1]
    n_outputs = weights.shape[2]
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if use_bf16 and x.is_cuda else contextlib.nullcontext())
    with autocast_ctx:
        idx_a = anchor_pairs_a.long(); idx_b = anchor_pairs_b.long()
        x_a = x[:, idx_a]; x_b = x[:, idx_b]
        d = x_a - x_b
        p = d / (T_soft + d.abs())
        ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
        sel_soft = F.softmax(ts / T_sel, dim=-1)
        # Hard forward via embedding_bag (sparse weights grad).
        index = ts.argmax(dim=-1)
        weights_flat = weights.view(n_tables * table_dim, n_outputs)
        table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
        flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
        n_bags = B * n_heads
        offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tables_per_head
        out_hard = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
        out_hard = out_hard.view(B, n_heads, n_outputs)
        # Phantom soft path for x/T_soft/T_sel grads. Zero-forward via STE.
        phantom = torch.einsum("btk,tko->bto", sel_soft, weights.detach())  # [B, T, n_outputs]
        phantom = phantom.view(B, n_heads, tables_per_head, n_outputs).sum(dim=2)
        out = out_hard + (phantom - phantom.detach())
    return out.to(weights.dtype)


# -------- Reference: Soft pipeline using THE SAME bit_matrix convention --------
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


# -------- Equivalence test --------
def make_test_module(input_dim, n_heads, tph, nap, n_outputs, device):
    n_tables = n_heads * tph
    rng = torch.Generator(device=device).manual_seed(0)
    a_rand = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b_rand = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    # Ensure a != b per slot
    b_rand = torch.where(b_rand == a_rand, (b_rand + 1) % input_dim, b_rand)
    weights = (torch.rand(n_tables, 1 << nap, n_outputs, generator=rng, device=device) - 0.5) * 0.002
    weights = weights.to(torch.float32).clone().requires_grad_(True)
    bm = _bit_matrix_msb(nap, device)
    return weights, a_rand.to(torch.int16), b_rand.to(torch.int16), bm


def gradient_check():
    device = torch.device("cuda")
    torch.manual_seed(0)
    B, input_dim, n_heads, tph, nap, n_outputs = 16, 64, 4, 8, 5, 12
    weights, ap_a, ap_b, bm = make_test_module(input_dim, n_heads, tph, nap, n_outputs, device)

    log_T_soft = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    T_soft = log_T_soft.exp(); T_sel = log_T_sel.exp()

    x_a = torch.randn(B, input_dim, device=device, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)

    # Reference
    out_a = soft_reference_forward(x_a, weights, ap_a, ap_b, bm, T_soft, T_sel,
                                   n_heads, tph, use_bf16=False)
    g_a_x, g_a_w, g_a_Ts, g_a_Tx = torch.autograd.grad(
        out_a.sum(), [x_a, weights, log_T_soft, log_T_sel],
        retain_graph=False, allow_unused=False
    )

    # Fast
    weights2 = weights.detach().clone().requires_grad_(True)
    log_T_soft2 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel2  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    out_b = _SoftFastFn.apply(x_b, weights2, ap_a, ap_b, bm,
                              log_T_soft2, log_T_sel2, n_heads, tph, False, False)
    g_b_x, g_b_w, g_b_Ts, g_b_Tx = torch.autograd.grad(
        out_b.sum(), [x_b, weights2, log_T_soft2, log_T_sel2],
        retain_graph=False, allow_unused=False
    )

    print(f"\n=== Equivalence check: ref vs soft-fast ===")
    print(f"  out abs|Δ|max  = {(out_a - out_b).abs().max().item():.2e}")
    print(f"  g_x  abs|Δ|max = {(g_a_x - g_b_x).abs().max().item():.2e}   ref|max| = {g_a_x.abs().max().item():.2e}")
    print(f"  g_w  abs|Δ|max = {(g_a_w - g_b_w).abs().max().item():.2e}   ref|max| = {g_a_w.abs().max().item():.2e}")
    print(f"  g_logTs abs|Δ| = {(g_a_Ts - g_b_Ts).abs().item():.2e}      ref = {g_a_Ts.item():.2e}")
    print(f"  g_logTx abs|Δ| = {(g_a_Tx - g_b_Tx).abs().item():.2e}      ref = {g_a_Tx.item():.2e}")

    # Hybrid (embedding_bag + phantom)
    weights3 = weights.detach().clone().requires_grad_(True)
    log_T_soft3 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel3  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x_c = x_a.detach().clone().requires_grad_(True)
    out_c = hybrid_forward(x_c, weights3, ap_a, ap_b, bm,
                           log_T_soft3.exp(), log_T_sel3.exp(), n_heads, tph, use_bf16=False)
    g_c_x, g_c_w, g_c_Ts, g_c_Tx = torch.autograd.grad(
        out_c.sum(), [x_c, weights3, log_T_soft3, log_T_sel3], retain_graph=False
    )
    print(f"\n=== Equivalence check: ref vs hybrid (embedding_bag + phantom) ===")
    print(f"  out abs|Δ|max  = {(out_a - out_c).abs().max().item():.2e}")
    print(f"  g_x  abs|Δ|max = {(g_a_x - g_c_x).abs().max().item():.2e}   ref|max| = {g_a_x.abs().max().item():.2e}")
    print(f"  g_w  abs|Δ|max = {(g_a_w - g_c_w).abs().max().item():.2e}   ref|max| = {g_a_w.abs().max().item():.2e}")
    print(f"  g_logTs abs|Δ| = {(g_a_Ts - g_c_Ts).abs().item():.2e}      ref = {g_a_Ts.item():.2e}")
    print(f"  g_logTx abs|Δ| = {(g_a_Tx - g_c_Tx).abs().item():.2e}      ref = {g_a_Tx.item():.2e}")

    # Narrow-Function variant
    weights4 = weights.detach().clone().requires_grad_(True)
    log_T_soft4 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel4  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x_d = x_a.detach().clone().requires_grad_(True)
    out_d = soft_then_embbag_forward(x_d, weights4, ap_a, ap_b, bm,
                                     log_T_soft4.exp(), log_T_sel4.exp(),
                                     n_heads, tph, use_bf16=False)
    g_d_x, g_d_w, g_d_Ts, g_d_Tx = torch.autograd.grad(
        out_d.sum(), [x_d, weights4, log_T_soft4, log_T_sel4], retain_graph=False
    )
    print(f"\n=== Equivalence check: ref vs narrow (soft→embbag custom Fn) ===")
    print(f"  out abs|Δ|max  = {(out_a - out_d).abs().max().item():.2e}")
    print(f"  g_x  abs|Δ|max = {(g_a_x - g_d_x).abs().max().item():.2e}   ref|max| = {g_a_x.abs().max().item():.2e}")
    print(f"  g_w  abs|Δ|max = {(g_a_w - g_d_w).abs().max().item():.2e}   ref|max| = {g_a_w.abs().max().item():.2e}")
    print(f"  g_logTs abs|Δ| = {(g_a_Ts - g_d_Ts).abs().item():.2e}      ref = {g_a_Ts.item():.2e}")
    print(f"  g_logTx abs|Δ| = {(g_a_Tx - g_d_Tx).abs().item():.2e}      ref = {g_a_Tx.item():.2e}")


# -------- Bench --------
def bench():
    device = torch.device("cuda")
    torch.manual_seed(0)
    B = 8 * 512
    CONFIGS = [
        dict(name="qk_joint",   input_dim=96, n_outputs=128, nap=6, tph=256, n_heads=6),
        dict(name="v_lut",      input_dim=96, n_outputs=32,  nap=8, tph=256, n_heads=6),
        dict(name="out_proj_L0",input_dim=192,n_outputs=96,  nap=6, tph=2048,n_heads=1),
        dict(name="out_proj_L2",input_dim=192,n_outputs=96,  nap=6, tph=1024,n_heads=1),
    ]
    print(f"\n=== Bench (B={B}, fwd+bwd, fp32 weights, bf16 autocast) ===")
    for cfg in CONFIGS:
        weights, ap_a, ap_b, bm = make_test_module(
            cfg["input_dim"], cfg["n_heads"], cfg["tph"], cfg["nap"], cfg["n_outputs"], device,
        )
        log_T_soft = torch.tensor(math.log(0.5), device=device, requires_grad=True)
        log_T_sel  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
        T_soft = log_T_soft.exp(); T_sel = log_T_sel.exp()
        x = torch.randn(B, cfg["input_dim"], device=device, requires_grad=True)
        target = torch.randn(B, cfg["n_heads"], cfg["n_outputs"], device=device)

        def run_ref():
            T_s = log_T_soft.exp(); T_x = log_T_sel.exp()
            return soft_reference_forward(x, weights, ap_a, ap_b, bm, T_s, T_x,
                                          cfg["n_heads"], cfg["tph"], use_bf16=True)
        run_ref_c = torch.compile(run_ref, dynamic=True)
        def run_hybrid():
            T_s = log_T_soft.exp(); T_x = log_T_sel.exp()
            return hybrid_forward(x, weights, ap_a, ap_b, bm, T_s, T_x,
                                  cfg["n_heads"], cfg["tph"], use_bf16=True)
        run_hybrid_c = torch.compile(run_hybrid, dynamic=True)
        def run_narrow():
            T_s = log_T_soft.exp(); T_x = log_T_sel.exp()
            return soft_then_embbag_forward(x, weights, ap_a, ap_b, bm, T_s, T_x,
                                            cfg["n_heads"], cfg["tph"], use_bf16=True)
        run_narrow_c = torch.compile(run_narrow, dynamic=True)

        for label, fn in [("reference (compile)", run_ref_c),
                          ("hybrid    (compile)", run_hybrid_c),
                          ("narrow-Fn (compile)", run_narrow_c)]:
            # Warmup
            for _ in range(8):
                out = fn(); loss = (out - target).square().sum(); loss.backward()
                x.grad = None; weights.grad = None
                if log_T_soft.grad is not None: log_T_soft.grad = None
                if log_T_sel.grad is not None:  log_T_sel.grad = None

            # Time forward and backward via CUDA events
            n_iter = 40
            fwd_evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        for _ in range(n_iter)]
            bwd_evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        for _ in range(n_iter)]

            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for i in range(n_iter):
                fwd_evts[i][0].record()
                out = fn()
                fwd_evts[i][1].record()
                loss = (out - target).square().sum()
                bwd_evts[i][0].record()
                loss.backward()
                bwd_evts[i][1].record()
                x.grad = None; weights.grad = None
                if log_T_soft.grad is not None: log_T_soft.grad = None
                if log_T_sel.grad is not None:  log_T_sel.grad = None
            torch.cuda.synchronize()
            fwd_ms = sum(s.elapsed_time(e) for s, e in fwd_evts) / n_iter
            bwd_ms = sum(s.elapsed_time(e) for s, e in bwd_evts) / n_iter
            tot_ms = fwd_ms + bwd_ms
            peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"  {cfg['name']:<13s}  {label:<26s}  fwd={fwd_ms:6.2f}  bwd={bwd_ms:6.2f}  total={tot_ms:6.2f} ms  peak={peak:7.1f} MB")


if __name__ == "__main__":
    gradient_check()
    bench()
