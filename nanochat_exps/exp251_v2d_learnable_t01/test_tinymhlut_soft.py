"""Verify TinyMultiHeadLut(backward_mode='soft') gradient parity against
SoftMultiHeadLUT(hard=True) reference, at exp251 LUT shapes, then bench.
"""
import math, contextlib
import torch
import torch.nn.functional as F

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def equiv_one(input_dim, n_heads, tph, nap, n_outputs, B, device, init_T=0.5):
    """Build TinyMHLut(soft) and SoftMHLut(hard=True) with same anchor pairs
    and weights, compare forward + gradients."""
    torch.manual_seed(0)
    # Build TinyMHLut(soft).
    tiny = TinyMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        weight_dtype=torch.float32, random_seed=0, device=device,
        backward_mode="soft", soft_score_temp=init_T, select_temp=init_T,
        learnable_temps=True, use_bf16=False,
    ).to(device)

    soft = SoftMultiHeadLUT(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        soft_score_temp=init_T, select_temp=init_T,
        gumbel=False, hard=True, learnable_temps=True,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        weight_dtype=torch.float32, random_seed=0, device=device,
        use_bf16=False, compile_forward=False,
    ).to(device)

    # Force same weights (both seeded the same, but anchor pair sampling may
    # differ — TinyMHLut uses TAPL, SoftMHLut uses get_balanced_anchor_pairs).
    # For an apples-to-apples test we just confirm both compute the same hard
    # forward + same soft backward on their OWN setup. We compare TinyMHLut
    # against a hand-rolled SoftMHLut-style reference using TinyMHLut's anchor
    # pairs and bit_matrix convention.
    return tiny, soft


def reference_soft_forward(x, weights, anchor_a, anchor_b, bit_matrix,
                            T_soft, T_sel, n_heads, tph, use_bf16=False):
    """Hand-rolled SoftMHLut(hard=True) using the SAME bit_matrix convention
    as TinyMHLut(soft) — so we can compare apples-to-apples on the same
    anchor pairs and weights."""
    B = x.shape[0]
    n_tables, nap = anchor_a.shape
    n_outputs = weights.shape[2]
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if use_bf16 and x.is_cuda else contextlib.nullcontext())
    with autocast_ctx:
        idx_a = anchor_a.long(); idx_b = anchor_b.long()
        rd = x[:, idx_a] - x[:, idx_b]
        p = rd / (T_soft + rd.abs())
        ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
        sel_soft = F.softmax(ts / T_sel, dim=-1)
        idx = sel_soft.argmax(dim=-1, keepdim=True)
        sel_hard = torch.zeros_like(sel_soft).scatter_(-1, idx, 1.0)
        sel = sel_hard - sel_soft.detach() + sel_soft
        out_t = torch.einsum("btk,tko->bto", sel, weights)
    out_t = out_t.to(weights.dtype)
    return out_t.view(B, n_heads, tph, n_outputs).sum(dim=2)


def equiv(device):
    """Compare TinyMHLut(soft) vs hand-rolled SoftMHLut-style reference, both
    using TinyMHLut's anchor pairs and bit_matrix."""
    torch.manual_seed(0)
    B, input_dim, n_heads, tph, nap, n_outputs = 16, 64, 4, 8, 6, 12
    tiny = TinyMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        weight_dtype=torch.float32, random_seed=0, device=device,
        backward_mode="soft", soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=False,
    ).to(device)

    x = torch.randn(B, input_dim, device=device, requires_grad=True)
    # Reference computed on TinyMHLut's own anchor pairs / bit_matrix / weights.
    weights_ref = tiny.weights.detach().clone().requires_grad_(True)
    log_Ts_ref = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    log_Tx_ref = torch.tensor(math.log(0.5), device=device, requires_grad=True)
    out_ref = reference_soft_forward(
        x, weights_ref, tiny.soft_anchor_a_long, tiny.soft_anchor_b_long,
        tiny.soft_bit_matrix, log_Ts_ref.exp(), log_Tx_ref.exp(),
        n_heads, tph, use_bf16=False,
    )
    g_ref = torch.autograd.grad(out_ref.sum(), [x, weights_ref, log_Ts_ref, log_Tx_ref])

    x2 = x.detach().clone().requires_grad_(True)
    out_new = tiny(x2)
    g_new = torch.autograd.grad(
        out_new.sum(),
        [x2, tiny.weights, tiny.log_soft_score_temp, tiny.log_select_temp],
    )

    print(f"\n=== Equivalence: TinyMHLut(soft) vs SoftMHLut-style ref (fp32, NAP={nap}) ===")
    print(f"  out abs|Δ|max  = {(out_ref - out_new).abs().max().item():.2e}")
    for name, a, b in zip(["g_x","g_w","g_logTs","g_logTx"], g_ref, g_new):
        diff = (a-b).abs().max().item()
        ref = a.abs().max().item() if a.dim() else abs(a.item())
        rel = diff / max(ref, 1e-12)
        print(f"  {name:8s}  abs|Δ|max = {diff:.2e}   ref|max| = {ref:.2e}   rel = {rel:.2e}")


def bench(device):
    B = 8 * 512
    CONFIGS = [
        dict(name="qk_joint",   input_dim=96, n_outputs=128, nap=6, tph=256, n_heads=6),
        dict(name="v_lut",      input_dim=96, n_outputs=32,  nap=8, tph=256, n_heads=6),
        dict(name="out_proj_L0",input_dim=192,n_outputs=96,  nap=6, tph=2048,n_heads=1),
        dict(name="out_proj_L2",input_dim=192,n_outputs=96,  nap=6, tph=1024,n_heads=1),
    ]
    print(f"\n=== Bench: SoftMHLut(compile) vs TinyMHLut(STE) vs TinyMHLut(soft), B={B} ===")
    for cfg in CONFIGS:
        torch.manual_seed(0)
        tiny_soft = TinyMultiHeadLut(
            input_dim=cfg["input_dim"], n_heads=cfg["n_heads"],
            n_outputs=cfg["n_outputs"], n_anchor_pairs=cfg["nap"],
            tables_per_head=cfg["tph"],
            weight_dtype=torch.float32, random_seed=0, device=device,
            backward_mode="soft", soft_score_temp=0.5, select_temp=0.5,
            learnable_temps=True, use_bf16=True,
        ).to(device)
        torch.manual_seed(0)
        tiny_ste = TinyMultiHeadLut(
            input_dim=cfg["input_dim"], n_heads=cfg["n_heads"],
            n_outputs=cfg["n_outputs"], n_anchor_pairs=cfg["nap"],
            tables_per_head=cfg["tph"],
            weight_dtype=torch.float32, random_seed=0, device=device,
            backward_mode="ste",
        ).to(device)
        torch.manual_seed(0)
        soft = SoftMultiHeadLUT(
            input_dim=cfg["input_dim"], n_heads=cfg["n_heads"],
            n_outputs=cfg["n_outputs"], n_anchor_pairs=cfg["nap"],
            tables_per_head=cfg["tph"],
            soft_score_temp=0.5, select_temp=0.5,
            gumbel=False, hard=True, learnable_temps=True,
            anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            weight_dtype=torch.float32, random_seed=0, device=device,
            use_bf16=True, compile_forward=True,
        ).to(device)

        x = torch.randn(B, cfg["input_dim"], device=device, requires_grad=True)
        target = torch.randn(B, cfg["n_heads"], cfg["n_outputs"], device=device)

        for label, mod in [("SoftMHLut(compile)", soft),
                           ("TinyMHLut(STE)",     tiny_ste),
                           ("TinyMHLut(soft)",    tiny_soft)]:
            for _ in range(8):
                out = mod(x); loss = (out - target).square().sum(); loss.backward()
                x.grad = None
                for p in mod.parameters(): p.grad = None
            n_iter = 30
            fwd = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
            bwd = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for i in range(n_iter):
                fwd[i][0].record(); out = mod(x); fwd[i][1].record()
                loss = (out - target).square().sum()
                bwd[i][0].record(); loss.backward(); bwd[i][1].record()
                x.grad = None
                for p in mod.parameters(): p.grad = None
            torch.cuda.synchronize()
            f_ms = sum(s.elapsed_time(e) for s, e in fwd) / n_iter
            b_ms = sum(s.elapsed_time(e) for s, e in bwd) / n_iter
            peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"  {cfg['name']:<13s}  {label:<20s}  fwd={f_ms:6.2f}  bwd={b_ms:6.2f}  total={f_ms+b_ms:6.2f} ms  peak={peak:7.1f} MB")


if __name__ == "__main__":
    dev = torch.device("cuda")
    equiv(dev)
    bench(dev)
