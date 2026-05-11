"""Phase 1: Python autograd Function that:
  - Forward: bit-pack signs of d, then embedding_bag (TinyMHLut-style fast path).
  - Backward dL/dweights: existing native `tiny_mhlut_backward_na1` (sparse scatter).
  - Backward dL/dsel_soft: bf16 cuBLAS via einsum.
  - Soft pipeline (recompute p, ts, sel_soft) + softmax_bw + d_p + rational sign + scatter
    to grad_x, plus dL/dlog_T_*: pure PyTorch (compile-friendly).

Tests: equivalence vs the same SoftMHLut(hard=True) reference used previously,
plus end-to-end bench against current SoftMHLut-compile.
"""
import math, time, contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.tiny_multi_head_lut import _get_tiny_mhlut_native


def _bit_matrix_msb(nap, device, dtype=torch.float32):
    n = 1 << nap
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


def _msb_powers(nap, device):
    return (1 << torch.arange(nap - 1, -1, -1, device=device, dtype=torch.int64))


class _SoftLUTNative(torch.autograd.Function):
    """Forward: TinyMHLut-style (sign-pack + embedding_bag).
    Backward dL/dweights: native `tiny_mhlut_backward_na1` (sparse scatter).
    Backward dL/dsel_soft: cuBLAS einsum.
    Soft path (dL/dx, dL/dT_*): standard PyTorch ops in backward."""

    @staticmethod
    def forward(ctx, x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                log_T_soft, log_T_sel, n_heads, tables_per_head, use_bf16):
        B, input_dim = x.shape
        n_tables, nap = anchor_pairs_a.shape
        table_dim = bit_matrix.shape[1]
        n_outputs = weights.shape[2]

        idx_a = anchor_pairs_a.long(); idx_b = anchor_pairs_b.long()
        x_a = x[:, idx_a]; x_b = x[:, idx_b]
        d = x_a - x_b
        bits = (d > 0).to(torch.int64)
        powers = _msb_powers(nap, x.device).view(1, 1, -1)
        index = (bits * powers).sum(dim=-1)                  # [B, T]

        # embedding_bag forward (gather + sum over tph per head)
        weights_flat = weights.view(n_tables * table_dim, n_outputs)
        table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
        flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
        n_bags = B * n_heads
        offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tables_per_head
        out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
        out = out_flat.view(B, n_heads, n_outputs)

        ctx.save_for_backward(x, d, weights, index, anchor_pairs_a, anchor_pairs_b,
                              bit_matrix, log_T_soft, log_T_sel)
        ctx.n_heads = n_heads
        ctx.tables_per_head = tables_per_head
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, d, weights, index, anchor_pairs_a, anchor_pairs_b, bit_matrix,
         log_T_soft, log_T_sel) = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tables_per_head
        B = d.shape[0]
        n_tables, nap = anchor_pairs_a.shape
        table_dim = bit_matrix.shape[1]
        n_outputs = weights.shape[2]
        input_dim = x.shape[1]

        # 1) dL/dweights via native sparse-scatter kernel.
        #    The native API expects lookup_alt_indices too (TAPL STE) — we pass
        #    the same as `index` since we don't use the alt carrier path.
        native = _get_tiny_mhlut_native()
        grad_out_w = grad_out.contiguous().to(weights.dtype)
        index64 = index.contiguous()
        if native is not None:
            grad_weights, _gm, _ga = native.tiny_mhlut_backward_na1(
                grad_out_w, weights, index64, index64, tph,  # alt = same idx (we ignore _gm/_ga)
            )
        else:
            # Pure-torch fallback
            grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
            flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
            flat_idx = (index + flat_offset.view(1, -1)).reshape(-1)
            grad_w_flat = torch.zeros(n_tables * table_dim, n_outputs,
                                      dtype=weights.dtype, device=weights.device)
            grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(weights.dtype))
            grad_weights = grad_w_flat.view(n_tables, table_dim, n_outputs)

        # 2-N) Soft path (recompute + soft bwd).
        T_soft = log_T_soft.exp()
        T_sel = log_T_sel.exp()

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if ctx.use_bf16 and d.is_cuda else contextlib.nullcontext()
        )
        with autocast_ctx:
            denom = T_soft + d.abs()
            p = d / denom
            ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
            z = ts / T_sel
            sel_soft = F.softmax(z, dim=-1)
            grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
            d_sel_soft = torch.einsum("bto,tko->btk", grad_pt, weights)
            sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
            d_z = sel_soft * (d_sel_soft - sum_term)
            d_ts = d_z / T_sel
            d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))
            d_d = d_p * (T_soft / (denom * denom))

        d_d_fp = d_d.to(torch.float32)
        d_fp = d.to(torch.float32)

        d_log_T_sel  = -(d_z.to(torch.float32) * z.to(torch.float32)).sum().reshape(())
        d_log_T_soft = -(d_d_fp * d_fp).sum().reshape(())

        grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=d.device)
        idx_a = anchor_pairs_a.long().unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        idx_b = anchor_pairs_b.long().unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        grad_x.scatter_add_(1, idx_a, d_d_fp.reshape(B, -1).to(x.dtype))
        grad_x.scatter_add_(1, idx_b, -d_d_fp.reshape(B, -1).to(x.dtype))

        return (grad_x, grad_weights, None, None, None,
                d_log_T_soft, d_log_T_sel, None, None, None)


# --- Reference: SoftMHLut(hard=True) using same bit_matrix convention ---
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
    torch.manual_seed(0)
    B, input_dim, n_heads, tph, nap, n_outputs = 16, 64, 4, 8, 5, 12
    weights, ap_a, ap_b, bm = make(input_dim, n_heads, tph, nap, n_outputs, device)

    log_T_soft = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x_a = torch.randn(B, input_dim, device=device, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)

    out_ref = soft_reference_forward(x_a, weights, ap_a, ap_b, bm, log_T_soft.exp(), log_T_sel.exp(),
                                     n_heads, tph, use_bf16=False)
    g_x_a, g_w_a, g_Ts_a, g_Tx_a = torch.autograd.grad(
        out_ref.sum(), [x_a, weights, log_T_soft, log_T_sel])

    weights2 = weights.detach().clone().requires_grad_(True)
    log_T_soft2 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel2  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    out_new = _SoftLUTNative.apply(x_b, weights2, ap_a, ap_b, bm,
                                   log_T_soft2, log_T_sel2, n_heads, tph, False)
    g_x_b, g_w_b, g_Ts_b, g_Tx_b = torch.autograd.grad(
        out_new.sum(), [x_b, weights2, log_T_soft2, log_T_sel2])

    print(f"\n=== Equivalence (small, fp32, native dW kernel) ===")
    print(f"  out abs|Δ|max  = {(out_ref - out_new).abs().max().item():.2e}")
    print(f"  g_x  abs|Δ|max = {(g_x_a - g_x_b).abs().max().item():.2e}   ref|max| = {g_x_a.abs().max().item():.2e}")
    print(f"  g_w  abs|Δ|max = {(g_w_a - g_w_b).abs().max().item():.2e}   ref|max| = {g_w_a.abs().max().item():.2e}")
    print(f"  g_logTs abs|Δ| = {(g_Ts_a - g_Ts_b).abs().item():.2e}      ref = {g_Ts_a.item():.2e}")
    print(f"  g_logTx abs|Δ| = {(g_Tx_a - g_Tx_b).abs().item():.2e}      ref = {g_Tx_a.item():.2e}")


def bench(device):
    B = 8 * 512
    CONFIGS = [
        dict(name="qk_joint",   input_dim=96, n_outputs=128, nap=6, tph=256, n_heads=6),
        dict(name="v_lut",      input_dim=96, n_outputs=32,  nap=8, tph=256, n_heads=6),
        dict(name="out_proj_L0",input_dim=192,n_outputs=96,  nap=6, tph=2048,n_heads=1),
        dict(name="out_proj_L2",input_dim=192,n_outputs=96,  nap=6, tph=1024,n_heads=1),
    ]
    print(f"\n=== Bench (B={B}, fp32 weights, bf16 autocast); fwd / bwd separated ===")
    for cfg in CONFIGS:
        weights, ap_a, ap_b, bm = make(cfg["input_dim"], cfg["n_heads"], cfg["tph"], cfg["nap"], cfg["n_outputs"], device)
        log_T_soft = torch.tensor(math.log(0.5), device=device, requires_grad=True)
        log_T_sel  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
        x = torch.randn(B, cfg["input_dim"], device=device, requires_grad=True)
        target = torch.randn(B, cfg["n_heads"], cfg["n_outputs"], device=device)

        def run_ref():
            T_s = log_T_soft.exp(); T_x = log_T_sel.exp()
            return soft_reference_forward(x, weights, ap_a, ap_b, bm, T_s, T_x,
                                          cfg["n_heads"], cfg["tph"], use_bf16=True)
        run_ref_c = torch.compile(run_ref, dynamic=True)
        def run_native():
            return _SoftLUTNative.apply(x, weights, ap_a, ap_b, bm, log_T_soft, log_T_sel,
                                        cfg["n_heads"], cfg["tph"], True)

        for label, fn in [("reference (compile)", run_ref_c),
                          ("native dW + python soft-bw", run_native)]:
            for _ in range(8):
                out = fn(); loss = (out - target).square().sum(); loss.backward()
                x.grad = None; weights.grad = None
                if log_T_soft.grad is not None: log_T_soft.grad = None
                if log_T_sel.grad is not None:  log_T_sel.grad = None

            n_iter = 30
            fwd_evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
            bwd_evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for i in range(n_iter):
                fwd_evts[i][0].record(); out = fn(); fwd_evts[i][1].record()
                loss = (out - target).square().sum()
                bwd_evts[i][0].record(); loss.backward(); bwd_evts[i][1].record()
                x.grad = None; weights.grad = None
                if log_T_soft.grad is not None: log_T_soft.grad = None
                if log_T_sel.grad is not None:  log_T_sel.grad = None
            torch.cuda.synchronize()
            fwd_ms = sum(s.elapsed_time(e) for s, e in fwd_evts) / n_iter
            bwd_ms = sum(s.elapsed_time(e) for s, e in bwd_evts) / n_iter
            peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"  {cfg['name']:<14s}  {label:<28s}  fwd={fwd_ms:6.2f}  bwd={bwd_ms:6.2f}  total={fwd_ms+bwd_ms:6.2f} ms  peak={peak:7.1f} MB")


if __name__ == "__main__":
    dev = torch.device("cuda")
    equiv_check(dev)
    bench(dev)
