"""R variant V2 — revised backward body with Python-level fusion hints:

  1. Pre-cast anchor indices to int64 and bit_matrix to working dtype outside
     the compile region (no per-call casts inside).
  2. Use `torch.ops.aten._softmax_backward_data` — the canonical softmax bw
     op, which compile recognizes and emits as one fused kernel instead of
     reconstructing it from {mul, sum, sub, mul}.
  3. Skip materializing `z = ts / T_sel` and `d_ts = d_z / T_sel` as separate
     [B, T, K] tensors — fold `/T_sel` into the final `d_d` and the scalar
     `grad_log_T_sel` expression. Saves ~2× [B, T, K] bf16 = ~6 GB at v_lut.
  4. Pre-compute `flat_offset` once at module init (or in forward) instead of
     recomputing inside every backward.
  5. Drop redundant `.to(weights.dtype)` casts when types match.
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
def _fwd_tiny_body(x, weights, anchor_a_long, anchor_b_long, powers,
                   n_heads, tph, table_dim):
    """Forward body — just bit-pack + embedding_bag."""
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
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
def _bw_recompute_body_v2(grad_out, x, weights, anchor_a_long, anchor_b_long,
                          bit_matrix, index, flat_offset, T_soft, T_sel,
                          n_heads, tph):
    """V2 backward — minimal intermediates, canonical softmax-bw op."""
    B, _, n_outputs = grad_out.shape
    n_tables = anchor_a_long.shape[0]
    K = bit_matrix.shape[1]
    input_dim = x.shape[1]
    w_dtype = weights.dtype

    # ----- Recompute forward intermediates (compile fuses these) -----
    d        = x[:, anchor_a_long] - x[:, anchor_b_long]
    denom    = T_soft + d.abs()
    p        = d / denom
    ts       = torch.einsum("btp,pk->btk", p, bit_matrix)
    sel_soft = F.softmax(ts / T_sel, dim=-1)

    # ----- Broadcast grad_out, dL/dsel_soft via cuBLAS -----
    grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt, weights)

    # ----- Canonical softmax backward (one fused op) -----
    # d_z = sel_soft * (d_sel_soft - sum(d_sel_soft * sel_soft, dim=-1, keepdim=True))
    d_z = torch.ops.aten._softmax_backward_data(d_sel_soft, sel_soft, -1, sel_soft.dtype)

    # ----- d_p via cuBLAS; /T_sel folded into d_d below -----
    d_p = torch.einsum("btk,pk->btp", d_z, bit_matrix)

    # ----- d_d (rational sign Jacobian + /T_sel folded) -----
    # d_ts = d_z / T_sel; d_p_eff = d_p / T_sel; d_d = d_p_eff * T_soft/denom^2
    inv_Ts = 1.0 / T_sel
    d_d = d_p * (inv_Ts * T_soft) / (denom * denom)

    # ----- Scalar grads (avoid materializing `z` and `d_ts`) -----
    # grad_log_T_sel = -sum(d_z * z) = -sum(d_z * ts) / T_sel
    grad_log_T_sel  = -(d_z * ts).sum() * inv_Ts
    grad_log_T_soft = -(d_d * d).sum()

    # ----- dL/dweights via scatter at saved flat_offset -----
    flat_idx = (index + flat_offset.view(1, -1)).reshape(-1)
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype, device=weights.device)
    grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(w_dtype))
    grad_weights = grad_w_flat.view(n_tables, K, n_outputs)

    # ----- dL/dx via two scatter_add (one per anchor side) -----
    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a_flat = anchor_a_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = anchor_b_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)

    return grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel


class _FastSoftLUT_R_V2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix_castd,
                powers, flat_offset, n_heads, tph, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        table_dim = bit_matrix_castd.shape[1]
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda else contextlib.nullcontext())
        with autocast_ctx:
            out, index = _fwd_tiny_body(x, weights, anchor_a_long, anchor_b_long,
                                        powers, n_heads, tph, table_dim)
        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix_castd, index, flat_offset,
                              log_T_soft, log_T_sel)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix, index,
         flat_offset, log_T_soft, log_T_sel) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda else contextlib.nullcontext())
        with autocast_ctx:
            grad_x, grad_w, grad_log_Ts, grad_log_Tx = _bw_recompute_body_v2(
                grad_out, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                index, flat_offset, T_soft, T_sel, ctx.n_heads, ctx.tph,
            )
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None)


def fast_R_v2_forward(x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                       powers, flat_offset, log_T_soft, log_T_sel,
                       n_heads, tph, use_bf16=True):
    return _FastSoftLUT_R_V2.apply(x, weights, log_T_soft, log_T_sel,
                                   anchor_a_long, anchor_b_long, bit_matrix,
                                   powers, flat_offset, n_heads, tph, use_bf16)


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


def make_precomputed(ap_a, ap_b, bit_matrix, n_tables, K, device, work_dtype=torch.bfloat16):
    """Pre-cast anchor pairs to int64, pre-cast bit_matrix to work_dtype,
    pre-compute powers and flat_offset. These are static-per-module values
    that the autograd Function can pass in saved/buffers."""
    anchor_a_long = ap_a.long().contiguous()
    anchor_b_long = ap_b.long().contiguous()
    bit_matrix_castd = bit_matrix.to(work_dtype).contiguous()
    nap = ap_a.shape[1]
    powers = (1 << torch.arange(nap - 1, -1, -1, device=device, dtype=torch.int64))
    flat_offset = (torch.arange(n_tables, device=device, dtype=torch.int64) * K).contiguous()
    return anchor_a_long, anchor_b_long, bit_matrix_castd, powers, flat_offset


def equiv():
    device = torch.device("cuda")
    torch.manual_seed(0)
    B, input_dim, n_heads, tph, nap, n_outputs = 16, 64, 4, 8, 6, 12
    weights, ap_a, ap_b, bm = make(input_dim, n_heads, tph, nap, n_outputs, device)
    n_tables = n_heads * tph
    K = 1 << nap
    aL, bL, bmW, powers, flat_offset = make_precomputed(ap_a, ap_b, bm, n_tables, K, device, work_dtype=torch.float32)
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
    out_new = fast_R_v2_forward(x_b, weights2, aL, bL, bmW, powers, flat_offset,
                                log_T_soft2, log_T_sel2, n_heads, tph, use_bf16=False)
    g_new = torch.autograd.grad(out_new.sum(), [x_b, weights2, log_T_soft2, log_T_sel2])
    print(f"\n=== Equivalence (R v2) ===")
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
        n_tables = cfg["n_heads"] * cfg["tph"]
        K = 1 << cfg["nap"]
        aL, bL, bmW, powers, flat_offset = make_precomputed(ap_a, ap_b, bm, n_tables, K, device)
        log_T_soft = torch.tensor(math.log(0.5), device=device, requires_grad=True)
        log_T_sel  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
        x = torch.randn(B, cfg["input_dim"], device=device, requires_grad=True)
        target = torch.randn(B, cfg["n_heads"], cfg["n_outputs"], device=device)

        def run_ref():
            return soft_reference_forward(x, weights, ap_a, ap_b, bm,
                                          log_T_soft.exp(), log_T_sel.exp(),
                                          cfg["n_heads"], cfg["tph"], use_bf16=True)
        run_ref_c = torch.compile(run_ref, dynamic=True)

        def run_R_v2():
            return fast_R_v2_forward(x, weights, aL, bL, bmW, powers, flat_offset,
                                      log_T_soft, log_T_sel,
                                      cfg["n_heads"], cfg["tph"], use_bf16=True)

        for label, fn in [("reference (compile)", run_ref_c),
                          ("fastR v2",             run_R_v2)]:
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
