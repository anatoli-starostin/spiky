"""Profile each component of the kernel-backed soft backward."""
import math, contextlib
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

def make(input_dim, n_heads, tph, nap, n_outputs, device):
    n_tables = n_heads * tph
    rng = torch.Generator(device=device).manual_seed(0)
    a = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b = torch.randint(0, input_dim, (n_tables, nap), generator=rng, device=device)
    b = torch.where(b == a, (b + 1) % input_dim, b)
    w = ((torch.rand(n_tables, 1 << nap, n_outputs, generator=rng, device=device) - 0.5) * 0.002)
    w = w.to(torch.float32)
    return w, a.to(torch.int16), b.to(torch.int16)


def main():
    dev = torch.device("cuda")
    B, input_dim, n_heads, tph, nap, n_outputs = 8*512, 96, 6, 256, 8, 32
    weights, ap_a, ap_b = make(input_dim, n_heads, tph, nap, n_outputs, dev)
    n_tables = n_heads * tph
    table_dim = 1 << nap

    # Build inputs that mimic backward state
    x = torch.randn(B, input_dim, device=dev)
    idx_a = ap_a.long(); idx_b = ap_b.long()
    d = (x[:, idx_a] - x[:, idx_b]).to(torch.bfloat16)  # [B, T, NAP]
    bits = (d > 0).to(torch.int64)
    powers = _msb_powers(nap, dev).view(1, 1, -1)
    index = (bits * powers).sum(dim=-1).contiguous()  # [B, T]

    grad_out = torch.randn(B, n_heads, n_outputs, device=dev)

    T_soft = 0.5; T_sel = 0.5

    # Warmup
    for _ in range(8):
        gw, _, _ = mgr.tiny_mhlut_backward_na1(grad_out, weights, index, index, tph)
    torch.cuda.synchronize()

    n_iter = 30

    # Time A: tiny_mhlut_backward_na1 (grad_weights)
    evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    for i in range(n_iter):
        evts[i][0].record()
        gw, _, _ = mgr.tiny_mhlut_backward_na1(grad_out, weights, index, index, tph)
        evts[i][1].record()
    torch.cuda.synchronize()
    t_dw = sum(s.elapsed_time(e) for s, e in evts) / n_iter
    print(f"  tiny_mhlut_backward_na1 (grad_weights):       {t_dw:6.2f} ms")

    # Time B: cuBLAS einsum d_sel_soft  (under bf16 autocast like the real bwd)
    evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    grad_pt_template = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
    for _ in range(8):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            d_sel_soft = torch.einsum("bto,tko->btk", grad_pt_template, weights)
    torch.cuda.synchronize()
    for i in range(n_iter):
        evts[i][0].record()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            d_sel_soft = torch.einsum("bto,tko->btk", grad_pt_template, weights)
        evts[i][1].record()
    torch.cuda.synchronize()
    t_dss = sum(s.elapsed_time(e) for s, e in evts) / n_iter
    print(f"  cuBLAS einsum d_sel_soft (bf16):              {t_dss:6.2f} ms")

    # Time C: new kernel for grad_x + temperature grads
    d_sel_soft_bf16 = d_sel_soft.contiguous().to(torch.bfloat16)
    d_buf_bf16 = d.contiguous()
    for _ in range(8):
        gx, gts, gtx = mgr.soft_lut_backward_grad_x(d_sel_soft_bf16, d_buf_bf16, ap_a, ap_b,
                                                    input_dim, T_soft, T_sel)
    torch.cuda.synchronize()
    evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    for i in range(n_iter):
        evts[i][0].record()
        gx, gts, gtx = mgr.soft_lut_backward_grad_x(d_sel_soft_bf16, d_buf_bf16, ap_a, ap_b,
                                                    input_dim, T_soft, T_sel)
        evts[i][1].record()
    torch.cuda.synchronize()
    t_kernel = sum(s.elapsed_time(e) for s, e in evts) / n_iter
    print(f"  new kernel (grad_x + grad_log_T_*):           {t_kernel:6.2f} ms")
    print(f"  --- sum of components:                        {t_dw + t_dss + t_kernel:6.2f} ms")
    print(f"  --- vs reference compile bwd v_lut: ~15 ms")


if __name__ == "__main__":
    main()
