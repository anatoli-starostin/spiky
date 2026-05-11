"""Approach (A): custom autograd Function with forward = embedding_bag,
backward body written as pure-PyTorch ops + @torch.compile.

The Function saves (sel_soft, weights, d, anchor_a, anchor_b, bit_matrix,
log_T_soft, log_T_sel) from forward so the backward body has everything
without recompute. The backward body is wholly pure-PyTorch — no scatter
operations done via custom ops, just torch.scatter_add and standard math —
so inductor can fuse the pointwise ops and tile the GEMMs.
"""
import math, contextlib
import torch
import torch.nn.functional as F


def _bit_matrix_msb(nap, device, dtype=torch.float32):
    n = 1 << nap
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


# Compiled backward body. Receives saved tensors + grad_out, returns the four grads.
@torch.compile(dynamic=True)
def _fast_soft_bw(grad_out, d, sel_soft, ts, weights, anchor_a, anchor_b,
                  bit_matrix, T_soft, T_sel, n_heads, tph):
    """Pure-PyTorch backward; will be compiled and fused by inductor."""
    B, _, n_outputs = grad_out.shape
    n_tables, NAP = anchor_a.shape
    K = 1 << NAP

    # Broadcast grad_out across tph and reshape to per-table.
    grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)

    # ---- dL/dsel_soft via einsum (cuBLAS tensor-cores) ----
    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(weights.dtype), weights)

    # ---- softmax backward → dL/dts and dL/dlog_T_sel ----
    sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
    d_z = sel_soft * (d_sel_soft - sum_term)
    d_ts = d_z / T_sel
    grad_log_T_sel = -(d_z * (ts / T_sel)).sum()

    # ---- dL/dp via einsum ----
    d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))

    # ---- dL/dd via rational sign Jacobian; dL/dlog_T_soft ----
    denom = T_soft + d.abs()
    d_d = d_p * (T_soft / (denom * denom))
    grad_log_T_soft = -(d_d * d).sum()

    return grad_pt, d_d, grad_log_T_soft, grad_log_T_sel


@torch.compile(dynamic=True)
def _scatter_grad_weights(grad_pt, index, n_tables, K, n_outputs, dtype, device):
    """dL/dweights via scatter at (table, index). Compiled separately so we can
    pre-allocate the output buffer outside (avoiding compile recapture on dtype)."""
    flat_offset = torch.arange(n_tables, device=device, dtype=index.dtype) * K
    flat_idx = (index + flat_offset[None, :]).reshape(-1)
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=dtype, device=device)
    grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(dtype))
    return grad_w_flat.view(n_tables, K, n_outputs)


@torch.compile(dynamic=True)
def _scatter_grad_x(d_d, anchor_a, anchor_b, B, input_dim, dtype, device):
    grad_x = torch.zeros(B, input_dim, dtype=dtype, device=device)
    idx_a_flat = anchor_a.long().unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = anchor_b.long().unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat = d_d.reshape(B, -1).to(dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)
    return grad_x


@torch.compile(dynamic=True)
def _fast_soft_fw_body(x, weights, anchor_a, anchor_b, bit_matrix,
                       T_soft, T_sel, n_heads, tph):
    """Compiled forward body — soft pipeline + embedding_bag."""
    B, _ = x.shape
    n_tables, nap = anchor_a.shape
    table_dim = bit_matrix.shape[1]
    n_outputs = weights.shape[2]
    idx_a = anchor_a.long(); idx_b = anchor_b.long()
    x_a = x[:, idx_a]; x_b = x[:, idx_b]
    d = x_a - x_b
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    sel_soft = F.softmax(ts / T_sel, dim=-1)
    index = ts.argmax(dim=-1)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    out = out_flat.view(B, n_heads, n_outputs)
    return out, d, sel_soft, ts, index


class _FastSoftLUT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_pairs_a, anchor_pairs_b, bit_matrix,
                n_heads, tables_per_head, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda else contextlib.nullcontext())
        with autocast_ctx:
            out, d, sel_soft, ts, index = _fast_soft_fw_body(
                x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                T_soft, T_sel, n_heads, tables_per_head)
        ctx.save_for_backward(d, sel_soft, ts, weights, index,
                              anchor_pairs_a, anchor_pairs_b, bit_matrix,
                              log_T_soft, log_T_sel)
        ctx.n_heads = n_heads
        ctx.tph = tables_per_head
        ctx.B = x.shape[0]
        ctx.input_dim = x.shape[1]
        ctx.x_dtype = x.dtype
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (d, sel_soft, ts, weights, index, anchor_a, anchor_b, bit_matrix,
         log_T_soft, log_T_sel) = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tph
        B = ctx.B
        input_dim = ctx.input_dim
        n_tables, NAP = anchor_a.shape
        K = 1 << NAP
        n_outputs = weights.shape[2]
        T_soft = log_T_soft.exp()
        T_sel = log_T_sel.exp()

        # Compute soft path gradients (pure-PyTorch, compiled).
        grad_pt, d_d, grad_log_T_soft, grad_log_T_sel = _fast_soft_bw(
            grad_out, d, sel_soft, ts, weights, anchor_a, anchor_b,
            bit_matrix, T_soft, T_sel, n_heads, tph,
        )
        # dL/dweights via scatter (sparse, matches STE semantics).
        grad_weights = _scatter_grad_weights(grad_pt, index, n_tables, K, n_outputs,
                                              weights.dtype, weights.device)
        # dL/dx via scatter at anchor positions.
        grad_x = _scatter_grad_x(d_d, anchor_a, anchor_b, B, input_dim,
                                  ctx.x_dtype, d.device)

        return (grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel,
                None, None, None, None, None, None)


def fast_soft_forward(x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                      log_T_soft, log_T_sel, n_heads, tables_per_head, use_bf16=True):
    return _FastSoftLUT.apply(x, weights, log_T_soft, log_T_sel,
                              anchor_pairs_a, anchor_pairs_b, bit_matrix,
                              n_heads, tables_per_head, use_bf16)


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
    out_new = fast_soft_forward(x_b, weights2, ap_a, ap_b, bm,
                                log_T_soft2, log_T_sel2, n_heads, tph, use_bf16=False)
    g_new = torch.autograd.grad(out_new.sum(), [x_b, weights2, log_T_soft2, log_T_sel2])

    print(f"\n=== Equivalence (compiled-bw approach, fp32, NAP={nap}) ===")
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

        def run_fast():
            return fast_soft_forward(x, weights, ap_a, ap_b, bm, log_T_soft, log_T_sel,
                                     cfg["n_heads"], cfg["tph"], use_bf16=True)

        for label, fn in [("reference (compile)", run_ref_c),
                          ("fast+compiledBW",     run_fast)]:
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
