"""Benchmark `soft_lut_backward_grad_x` kernel-backed soft LUT vs SoftMHLut(compile)."""
import math, time, contextlib
import torch
import torch.nn.functional as F

import lutorch_cuda
mgr = lutorch_cuda.get_lutorch_manager()


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
        out = out_flat.view(B, n_heads, n_outputs)
        d_save = d.to(torch.bfloat16) if use_bf16 and x.is_cuda else d.contiguous()
        ctx.save_for_backward(x, d_save, weights, index, anchor_pairs_a, anchor_pairs_b,
                              log_T_soft, log_T_sel)
        ctx.n_heads = n_heads; ctx.tables_per_head = tables_per_head
        ctx.use_bf16 = use_bf16; ctx.input_dim = input_dim
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, d_save, weights, index, ap_a, ap_b,
         log_T_soft, log_T_sel) = ctx.saved_tensors
        n_heads = ctx.n_heads; tph = ctx.tables_per_head
        B = x.shape[0]; n_tables, nap = ap_a.shape; n_outputs = weights.shape[2]
        T_soft = float(log_T_soft.detach().exp().item())
        T_sel  = float(log_T_sel.detach().exp().item())

        grad_weights, _gm, _ga = mgr.tiny_mhlut_backward_na1(
            grad_out.contiguous().to(weights.dtype),
            weights, index.contiguous(), index.contiguous(), tph,
        )
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda else contextlib.nullcontext())
        with autocast_ctx:
            grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
            d_sel_soft = torch.einsum("bto,tko->btk", grad_pt, weights)

        d_buf = d_save if d_save.dtype == d_sel_soft.dtype else d_save.to(d_sel_soft.dtype)
        grad_x_fp32, d_log_T_soft, d_log_T_sel = mgr.soft_lut_backward_grad_x(
            d_sel_soft.contiguous(), d_buf.contiguous(),
            ap_a, ap_b, ctx.input_dim, T_soft, T_sel,
        )
        return (grad_x_fp32.to(x.dtype), grad_weights, None, None,
                d_log_T_soft, d_log_T_sel, None, None, None)


def soft_reference_forward(x, weights, ap_a, ap_b, bm, T_soft, T_sel, n_heads, tph, use_bf16=False):
    B = x.shape[0]; n_tables, nap = ap_a.shape; n_outputs = weights.shape[2]
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if use_bf16 and x.is_cuda else contextlib.nullcontext())
    with autocast_ctx:
        idx_a = ap_a.long(); idx_b = ap_b.long()
        rd = x[:, idx_a] - x[:, idx_b]
        p = rd / (T_soft + rd.abs())
        ts = torch.einsum("btp,pk->btk", p, bm.to(p.dtype))
        sel_soft = F.softmax(ts / T_sel, dim=-1)
        idx = sel_soft.argmax(dim=-1, keepdim=True)
        sel_hard = torch.zeros_like(sel_soft).scatter_(-1, idx, 1.0)
        sel = sel_hard - sel_soft.detach() + sel_soft
        out_t = torch.einsum("btk,tko->bto", sel, weights)
    return out_t.to(weights.dtype).view(B, n_heads, tph, n_outputs).sum(dim=2)


def make(input_dim, n_heads, tph, nap, n_outputs, device, weight_dtype=torch.float32):
    n_tables = n_heads * tph
    rng = torch.Generator(device=device).manual_seed(0)
    a = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b = torch.where(b == a, (b + 1) % input_dim, b)
    w = ((torch.rand(n_tables, 1 << nap, n_outputs, generator=rng, device=device) - 0.5) * 0.002)
    w = w.to(weight_dtype).clone().requires_grad_(True)
    bm = _bit_matrix_msb(nap, device, dtype=weight_dtype)
    return w, a.to(torch.int16), b.to(torch.int16), bm


def main():
    dev = torch.device("cuda")
    B = 8 * 512
    CONFIGS = [
        dict(name="qk_joint",   input_dim=96, n_outputs=128, nap=6, tph=256, n_heads=6),
        dict(name="v_lut",      input_dim=96, n_outputs=32,  nap=8, tph=256, n_heads=6),
        dict(name="out_proj_L0",input_dim=192,n_outputs=96,  nap=6, tph=2048,n_heads=1),
        dict(name="out_proj_L2",input_dim=192,n_outputs=96,  nap=6, tph=1024,n_heads=1),
    ]
    print(f"=== Bench (B={B}, fp32 weights, bf16 autocast) ===")
    for cfg in CONFIGS:
        weights, ap_a, ap_b, bm = make(cfg["input_dim"], cfg["n_heads"], cfg["tph"], cfg["nap"], cfg["n_outputs"], dev)
        log_T_soft = torch.tensor(math.log(0.5), device=dev, requires_grad=True)
        log_T_sel  = torch.tensor(math.log(0.5), device=dev, requires_grad=True)
        x = torch.randn(B, cfg["input_dim"], device=dev, requires_grad=True)
        target = torch.randn(B, cfg["n_heads"], cfg["n_outputs"], device=dev)

        def run_ref():
            return soft_reference_forward(x, weights, ap_a, ap_b, bm,
                                          log_T_soft.exp(), log_T_sel.exp(),
                                          cfg["n_heads"], cfg["tph"], use_bf16=True)
        run_ref_c = torch.compile(run_ref, dynamic=True)

        def run_kernel():
            return _SoftLUTKernel.apply(x, weights, ap_a, ap_b, log_T_soft, log_T_sel,
                                        cfg["n_heads"], cfg["tph"], True)

        for label, fn in [("reference (compile)", run_ref_c), ("kernel-backed", run_kernel)]:
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
    main()
