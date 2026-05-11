"""Pure-PyTorch approach: phantom-einsum custom Function for the soft backward.

Forward: embedding_bag (TinyMHLut-fast). Soft pipeline (p, ts, sel_soft)
computed in compile-friendly PyTorch.

Phantom: tiny custom autograd Function that returns zeros in forward but,
in backward, computes dL/dsel_soft = einsum(grad, weights^T). No grad to
weights from phantom — weights gradient flows only through embedding_bag's
native sparse backward.

Net: same gradients as reference SoftMHLut(hard=True), but forward replaces
the dominant einsum with embedding_bag. All compile-friendly except the
phantom's two-op forward/backward.
"""
import math, time, contextlib
import torch
import torch.nn.functional as F


def _bit_matrix_msb(nap, device, dtype=torch.float32):
    n = 1 << nap
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


class _PhantomEinsum(torch.autograd.Function):
    """forward: returns zeros of shape [B, n_heads, O].
       backward: dL/dsel_soft = einsum("bto,tko->btk", grad_pt, weights_det)
                 dL/dweights_det = None  (weights gradient handled by emb_bag)
    """
    @staticmethod
    def forward(ctx, sel_soft, weights_det, n_heads, tph, out_dtype):
        B, T, K = sel_soft.shape
        O = weights_det.shape[2]
        ctx.save_for_backward(sel_soft, weights_det)
        ctx.n_heads = n_heads
        ctx.tph = tph
        # Zero tensor of correct shape and dtype.
        return torch.zeros(B, n_heads, O, dtype=out_dtype, device=sel_soft.device)

    @staticmethod
    def backward(ctx, grad_out):
        sel_soft, weights_det = ctx.saved_tensors
        B, T, K = sel_soft.shape
        O = weights_det.shape[2]
        # Broadcast grad_out across tph; cuBLAS handles the einsum.
        grad_pt = grad_out.unsqueeze(2).expand(B, ctx.n_heads, ctx.tph, O).reshape(B, T, O)
        d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(weights_det.dtype), weights_det)
        return d_sel_soft, None, None, None, None


def phantom_forward(x, weights, anchor_pairs_a, anchor_pairs_b, bit_matrix,
                    T_soft, T_sel, n_heads, tables_per_head, use_bf16=False):
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
    # Hard forward via embedding_bag — native autograd → sparse dL/dweights.
    # `index` from ts.argmax (sign-bit packed). Compile may identify this as
    # equal to `d > 0` bit-pack — both produce the same int.
    index = ts.argmax(dim=-1)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tables_per_head
    out_hard = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    out_hard = out_hard.view(B, n_heads, n_outputs)
    # Phantom: zero in forward, einsum in backward. weights.detach() so
    # phantom does NOT contribute to grad_weights.
    phantom = _PhantomEinsum.apply(sel_soft, weights.detach(), n_heads, tables_per_head, out_hard.dtype)
    return out_hard + phantom


# --- Reference: SoftMHLut(hard=True) ---
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
                                     log_T_soft.exp(), log_T_sel.exp(), n_heads, tph, use_bf16=False)
    g_ref = torch.autograd.grad(out_ref.sum(), [x_a, weights, log_T_soft, log_T_sel])

    weights2 = weights.detach().clone().requires_grad_(True)
    log_T_soft2 = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_T_sel2  = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    out_new = phantom_forward(x_b, weights2, ap_a, ap_b, bm,
                              log_T_soft2.exp(), log_T_sel2.exp(), n_heads, tph, use_bf16=False)
    g_new = torch.autograd.grad(out_new.sum(), [x_b, weights2, log_T_soft2, log_T_sel2])

    print(f"\n=== Equivalence (phantom approach, fp32, NAP={nap}) ===")
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
        def run_phantom():
            return phantom_forward(x, weights, ap_a, ap_b, bm,
                                   log_T_soft.exp(), log_T_sel.exp(),
                                   cfg["n_heads"], cfg["tph"], use_bf16=True)
        run_phantom_c = torch.compile(run_phantom, dynamic=True)

        for label, fn in [("reference (compile)", run_ref_c),
                          ("phantom   (compile)", run_phantom_c)]:
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
