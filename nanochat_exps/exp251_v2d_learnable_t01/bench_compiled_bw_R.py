"""Approach (A) variant R: forward = embedding_bag only (TinyMHLut-tiny),
backward body recomputes the soft pipeline inside @torch.compile.

Forward saves only inputs (x, weights, anchor pairs, etc.) — no [B, T, K]
activations. Backward recomputes p, ts, sel_soft inside a single compiled
function that produces all four gradients.
"""
import math, contextlib
import torch
import torch.nn.functional as F


def _bit_matrix_msb(nap, device, dtype=torch.float32):
    n = 1 << nap
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


@torch.compile
def _fwd_tiny_body(x, weights, anchor_a, anchor_b, bit_matrix, powers,
                   T_soft, T_sel, n_heads, tph):
    """Forward body: bit-pack the sign of d, then embedding_bag. No soft
    pipeline computed in forward. Returns out and the small `index` tensor."""
    B, _ = x.shape
    n_tables, nap = anchor_a.shape
    table_dim = bit_matrix.shape[1]
    n_outputs = weights.shape[2]
    idx_a = anchor_a.long(); idx_b = anchor_b.long()
    x_a = x[:, idx_a]; x_b = x[:, idx_b]
    d = x_a - x_b
    # Bit-packed argmax index from sign bits — no soft pipeline needed for index.
    bits = (d > 0).to(torch.int64)
    index = (bits * powers.view(1, 1, -1)).sum(dim=-1)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs), index


@torch.compile
def _bw_recompute_body(grad_out, x, weights, anchor_a, anchor_b, bit_matrix,
                        index, T_soft, T_sel, n_heads, tph):
    """Single compiled backward body. Recomputes p, ts, sel_soft and produces
    all four gradients in one fused graph (modulo cuBLAS einsums and scatters)."""
    B, _, n_outputs = grad_out.shape
    n_tables = anchor_a.shape[0]
    K = bit_matrix.shape[1]            # = 1 << NAP; concrete tensor dim
    input_dim = x.shape[1]

    # ===== Recompute forward intermediates =====
    idx_a = anchor_a.long(); idx_b = anchor_b.long()
    d        = x[:, idx_a] - x[:, idx_b]
    denom    = T_soft + d.abs()
    p        = d / denom
    ts       = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

    # ===== Broadcast grad_out for per-table grad =====
    grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)

    # ===== dL/dsel_soft via cuBLAS GEMM =====
    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(weights.dtype), weights)

    # ===== softmax backward → d_z, d_ts =====
    sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
    d_z = sel_soft * (d_sel_soft - sum_term)
    d_ts = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    # ===== d_p via cuBLAS GEMM =====
    d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))

    # ===== d_d via rational sign Jacobian =====
    d_d = d_p * (T_soft / (denom * denom))
    grad_log_T_soft = -(d_d * d).sum()

    # ===== dL/dweights via scatter at (table, index) =====
    flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K
    flat_idx = (index + flat_offset[None, :]).reshape(-1)
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=weights.dtype, device=weights.device)
    grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(weights.dtype))
    grad_weights = grad_w_flat.view(n_tables, K, n_outputs)

    # ===== dL/dx via scatter-add at anchor positions =====
    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a_flat = idx_a.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = idx_b.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)

    return grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel


class _FastSoftLUT_R(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_pairs_a, anchor_pairs_b, bit_matrix,
                n_heads, tables_per_head, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        nap = anchor_pairs_a.shape[1]
        powers = (1 << torch.arange(nap - 1, -1, -1, device=x.device, dtype=torch.int64))
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda else contextlib.nullcontext())
        with autocast_ctx:
            out, index = _fwd_tiny_body(x, weights, anchor_pairs_a, anchor_pairs_b,
                                        bit_matrix, powers, T_soft, T_sel, n_heads, tables_per_head)
        # Save only small things: x, weights (param ref), anchor pairs (buffer),
        # bit_matrix (buffer), index, log_T_* (params).
        ctx.save_for_backward(x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                              index, log_T_soft, log_T_sel)
        ctx.n_heads = n_heads
        ctx.tph = tables_per_head
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a, anchor_b, bit_matrix, index,
         log_T_soft, log_T_sel) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda else contextlib.nullcontext())
        with autocast_ctx:
            grad_x, grad_w, grad_log_Ts, grad_log_Tx = _bw_recompute_body(
                grad_out, x, weights, anchor_a, anchor_b, bit_matrix, index,
                T_soft, T_sel, ctx.n_heads, ctx.tph,
            )
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None)


def fast_R_forward(x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                    log_T_soft, log_T_sel, n_heads, tph, use_bf16=True):
    return _FastSoftLUT_R.apply(x, weights, log_T_soft, log_T_sel,
                                anchor_pairs_a, anchor_pairs_b, bit_matrix,
                                n_heads, tph, use_bf16)


def soft_reference_forward(x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                           T_soft, T_sel, n_heads, tables_per_head, use_bf16=False):
    B = x.shape[0]
    n_tables, nap = anchor_pairs_a.shape
    n_outputs = weights.shape[2]
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if use_bf16 and x.is_cuda else contextlib.nullcontext())
    with autocast_ctx:
        idx_a = anchor_pairs_a.long(); idx_b = anchor_pairs_b.long()
        rd = x[:, idx_a] - x[:, idx_b]
        p = rd / (T_soft + rd.abs())
        ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
        sel_soft = F.softmax(ts / T_sel, dim=-1)
        idx = sel_soft.argmax(dim=-1, keepdim=True)
        sel_hard = torch.zeros_like(sel_soft).scatter_(-1, idx, 1.0)
        sel = sel_hard - sel_soft.detach() + sel_soft
        out_t = torch.einsum("btk,tko->bto", sel, weights)
    out_t = out_t.to(weights.dtype)
    return out_t.view(B, n_heads, tables_per_head, n_outputs).sum(dim=2)


def make(input_dim, n_heads, tph, nap, n_outputs, device):
    n_tables = n_heads * tph
    rng = torch.Generator(device=device).manual_seed(0)
    a = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b = torch.where(b == a, (b + 1) % input_dim, b)
    w = ((torch.rand(n_tables, 1 << nap, n_outputs, generator=rng, device=device) - 0.5) * 0.002)
    w = w.to(torch.float32).clone().requires_grad_(True)
    bm = _bit_matrix_msb(nap, device, dtype=torch.float32)
    return w, a.to(torch.int16), b.to(torch.int16), bm


def equiv():
    device = torch.device("cuda")
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
    out_new = fast_R_forward(x_b, weights2, ap_a, ap_b, bm,
                              log_T_soft2, log_T_sel2, n_heads, tph, use_bf16=False)
    g_new = torch.autograd.grad(out_new.sum(), [x_b, weights2, log_T_soft2, log_T_sel2])
    print(f"\n=== Equivalence (R variant) ===")
    print(f"  out abs|Δ|max  = {(out_ref - out_new).abs().max().item():.2e}")
    for name, a, b in zip(["g_x","g_w","g_logTs","g_logTx"], g_ref, g_new):
        diff = (a-b).abs().max().item()
        ref = a.abs().max().item() if a.dim() else abs(a.item())
        rel = diff / max(ref, 1e-12)
        print(f"  {name:8s}  abs|Δ|max = {diff:.2e}   ref|max| = {ref:.2e}   rel = {rel:.2e}")


def bench():
    device = torch.device("cuda")
    B = 8 * 512
    CONFIGS = [
        dict(name="qk_joint",   input_dim=96, n_outputs=128, nap=6, tph=256, n_heads=6),
        dict(name="v_lut",      input_dim=96, n_outputs=32,  nap=8, tph=256, n_heads=6),
        dict(name="out_proj_L0",input_dim=192,n_outputs=96,  nap=6, tph=2048,n_heads=1),
        dict(name="out_proj_L2",input_dim=192,n_outputs=96,  nap=6, tph=1024,n_heads=1),
    ]
    print(f"\n=== Bench (B={B}, fp32 weights, bf16 autocast) ===")
    for cfg in CONFIGS:
        weights, ap_a, ap_b, bm = make(cfg["input_dim"], cfg["n_heads"], cfg["tph"],
                                       cfg["nap"], cfg["n_outputs"], device)
        log_T_soft = torch.tensor(math.log(0.5), device=device, requires_grad=True)
        log_T_sel  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
        x = torch.randn(B, cfg["input_dim"], device=device, requires_grad=True)
        target = torch.randn(B, cfg["n_heads"], cfg["n_outputs"], device=device)

        def run_ref():
            return soft_reference_forward(x, weights, ap_a, ap_b, bm,
                                          log_T_soft.exp(), log_T_sel.exp(),
                                          cfg["n_heads"], cfg["tph"], use_bf16=True)
        run_ref_c = torch.compile(run_ref, dynamic=True)

        def run_R():
            return fast_R_forward(x, weights, ap_a, ap_b, bm, log_T_soft, log_T_sel,
                                   cfg["n_heads"], cfg["tph"], use_bf16=True)

        for label, fn in [("reference (compile)", run_ref_c),
                          ("fastR (compileBW)",   run_R)]:
            for _ in range(8):
                out = fn(); loss = (out - target).square().sum(); loss.backward()
                x.grad = None; weights.grad = None
                if log_T_soft.grad is not None: log_T_soft.grad = None
                if log_T_sel.grad is not None:  log_T_sel.grad = None
            n_iter = 30
            fwd = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
            bwd = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for i in range(n_iter):
                fwd[i][0].record(); out = fn(); fwd[i][1].record()
                loss = (out - target).square().sum()
                bwd[i][0].record(); loss.backward(); bwd[i][1].record()
                x.grad = None; weights.grad = None
                if log_T_soft.grad is not None: log_T_soft.grad = None
                if log_T_sel.grad is not None:  log_T_sel.grad = None
            torch.cuda.synchronize()
            f_ms = sum(s.elapsed_time(e) for s, e in fwd) / n_iter
            b_ms = sum(s.elapsed_time(e) for s, e in bwd) / n_iter
            peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"  {cfg['name']:<13s}  {label:<22s}  fwd={f_ms:6.2f}  bwd={b_ms:6.2f}  total={f_ms+b_ms:6.2f} ms  peak={peak:7.1f} MB")


if __name__ == "__main__":
    equiv()
    bench()
