"""Bench: custom autograd Function — gather-forward + sparse-weight-backward.

Replaces the original (gumbel=False, hard=True) inner ops:

    sel_soft = F.softmax(ts / T_sel, dim=-1)              # [B, T, K]
    sel_hard = scatter(zeros_like(sel_soft), argmax)       # [B, T, K] (sparse one-hot)
    sel = sel_hard - sel_soft.detach() + sel_soft           # [B, T, K]
    out = einsum("btk,tko->bto", sel, weights)              # [B, T, O]

with a single autograd Function that:
    forward:  out = weights[t, argmax(ts)[b,t], :]          # gather
    backward: dL/dweights = scatter_add (sparse, only argmax row, matches original)
              dL/dsel_soft = einsum("bto,tko->btk", grad_out, weights)

Saves: materialising sel_hard and sel as [B, T, K] tensors (each ~3.2 GB at v_lut shape).
"""
import time
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = "cuda"
torch.manual_seed(0)


class STEHardSelectFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sel_soft, weights, index):
        # sel_soft: [B, T, K] (kept for backward grad path)
        # weights:  [T, K, O]
        # index:    [B, T]    (no grad; from argmax of ts)
        ctx.save_for_backward(sel_soft, weights, index)
        B, T = index.shape
        t_idx = torch.arange(T, device=weights.device).unsqueeze(0).expand(B, -1)
        out = weights[t_idx, index, :]                     # [B, T, O], gather
        return out

    @staticmethod
    def backward(ctx, grad_out):
        sel_soft, weights, index = ctx.saved_tensors
        # dL/dsel_soft via einsum (smooth path through softmax)
        d_sel = torch.einsum("bto,tko->btk", grad_out, weights)
        # dL/dweights via sparse scatter_add at the picked rows (one-hot mask)
        d_w = torch.zeros_like(weights)
        B, T = index.shape
        t_idx = torch.arange(T, device=weights.device).unsqueeze(0).expand(B, -1)
        d_w.index_put_(
            (t_idx.flatten(), index.flatten()),
            grad_out.reshape(-1, grad_out.shape[-1]),
            accumulate=True,
        )
        return d_sel, d_w, None


def gather_ste_forward(self, x):
    import contextlib
    B = x.shape[0]
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if (self.use_bf16 and x.is_cuda)
        else contextlib.nullcontext()
    )
    T_soft, T_sel = self._temps()
    with autocast_ctx:
        x_a = x[:, self.anchor_pairs_a]
        x_b = x[:, self.anchor_pairs_b]
        rd = x_a - x_b
        p = rd / (T_soft + rd.abs())
        ts = torch.einsum("btp,pk->btk", p, self.bit_matrix.to(p.dtype))
        sel_soft = F.softmax(ts / T_sel, dim=-1)
        index = ts.argmax(dim=-1)
        out_t = STEHardSelectFn.apply(sel_soft, self.weights, index)
    out_t = out_t.to(self.weights.dtype)
    return out_t.view(B, self.n_heads, self.tables_per_head, self.n_outputs).sum(dim=2)


CONFIGS = [
    dict(name="qk_joint",   input_dim=96, n_outputs=128, nap=6, tph=256, n_heads=6),
    dict(name="v_lut",      input_dim=96, n_outputs=32,  nap=8, tph=256, n_heads=6),
    dict(name="out_proj_L0",input_dim=192,n_outputs=96,  nap=6, tph=2048,n_heads=1),
    dict(name="out_proj_L2",input_dim=192,n_outputs=96,  nap=6, tph=1024,n_heads=1),
]
B_TOK = 8 * 512


def make(cfg):
    return SoftMultiHeadLUT(
        input_dim=cfg["input_dim"],
        n_outputs=cfg["n_outputs"],
        n_anchor_pairs=cfg["nap"],
        tables_per_head=cfg["tph"],
        n_heads=cfg["n_heads"],
        device=DEVICE,
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=0.001,
        soft_score_temp=0.5,
        select_temp=0.5,
        gumbel=False,
        hard=True,
        learnable_temps=True,
        use_bf16=True,
        compile_forward=False,
    ).to(DEVICE)


def grads(mod, x):
    out = mod(x)
    g = torch.autograd.grad(
        out.sum(),
        [x, mod.weights, mod.log_soft_score_temp, mod.log_select_temp],
        retain_graph=False,
    )
    return out.detach(), g


print("\n=== Equivalence check (eager) ===")
mod_orig = make(CONFIGS[1])  # v_lut
mod_new  = make(CONFIGS[1])
mod_new.load_state_dict(mod_orig.state_dict())
mod_new.forward = types.MethodType(gather_ste_forward, mod_new)

torch.manual_seed(42)
x = torch.randn(B_TOK, CONFIGS[1]["input_dim"], device=DEVICE, requires_grad=True)
out_a, g_a = grads(mod_orig, x)
torch.manual_seed(42)
x = torch.randn(B_TOK, CONFIGS[1]["input_dim"], device=DEVICE, requires_grad=True)
out_b, g_b = grads(mod_new, x)

print(f"  out match    max|Δ| = {(out_a - out_b).abs().max():.2e}")
for name, (a, b) in zip(["dx","dw","dT_soft","dT_sel"], zip(g_a, g_b)):
    rel = (a - b).abs().max() / (a.abs().max() + 1e-12)
    print(f"  grad {name:8s}  max|Δ| = {(a-b).abs().max():.2e}   rel = {rel:.2e}")


def bench(mod, x, label, n_warm=8, n_iter=40):
    target = torch.randn(x.shape[0], mod.n_heads, mod.n_outputs, device=DEVICE)
    for _ in range(n_warm):
        out = mod(x)
        loss = (out - target).square().sum()
        loss.backward()
        x.grad = None
        for p in mod.parameters():
            p.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_iter):
        out = mod(x)
        loss = (out - target).square().sum()
        loss.backward()
        x.grad = None
        for p in mod.parameters():
            p.grad = None
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter * 1000
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"  {label:<30s} {dt:6.2f} ms (fwd+bwd)  peak={peak:7.1f} MB")
    return dt, peak


for cfg in CONFIGS:
    print(f"\n=== {cfg['name']}  in={cfg['input_dim']} out={cfg['n_outputs']} "
          f"nap={cfg['nap']} tph={cfg['tph']} H={cfg['n_heads']}  B={B_TOK} ===")
    mod = make(cfg)
    mod.forward = torch.compile(mod.forward, dynamic=True)
    x = torch.randn(B_TOK, cfg["input_dim"], device=DEVICE, requires_grad=True)
    t_orig, m_orig = bench(mod, x, "original (compiled)")

    mod = make(cfg)
    mod.forward = torch.compile(types.MethodType(gather_ste_forward, mod), dynamic=True)
    x = torch.randn(B_TOK, cfg["input_dim"], device=DEVICE, requires_grad=True)
    t_new, m_new = bench(mod, x, "gather+custom-bw (compiled)")
    print(f"  -> speed: {(t_orig - t_new)/t_orig*100:+.1f}%  memory: {(m_new - m_orig)/m_orig*100:+.1f}%")
