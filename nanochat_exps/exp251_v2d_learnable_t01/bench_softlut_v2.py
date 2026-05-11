"""A/B benchmark: original SoftMultiHeadLUT.forward vs an output-STE forward.

Hypothesis: the original code allocates 3 large `[B, n_tables, 2^nap]` tensors
(`sel_soft`, `sel_hard`, combined `sel`) just to compute a one-einsum forward
where forward value = sel_hard @ weights (a gather). Refactoring to STE on
output instead of selector keeps mathematics identical but
- only allocates `sel_soft` (1 tensor instead of 3)
- replaces `sel_hard @ weights` with cheap gather
- backward path through sel_soft @ weights is unchanged

We benchmark on the worst case from the model (v_lut: nap=8, tph=256, H=6).
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT, _bit_matrix
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = "cuda"
torch.manual_seed(0)


def output_ste_forward(self, x):
    """Replacement for SoftMultiHeadLUT.forward in (gumbel=False, hard=True) mode.

    Algebraically identical to original. Memory savings: avoids materialising
    `sel_hard` (zeros_like + scatter) and the combined `sel` tensor.
    """
    import contextlib
    B = x.shape[0]
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if (self.use_bf16 and x.is_cuda)
        else contextlib.nullcontext()
    )
    T_soft, T_sel = self._temps()
    n_outputs = self.n_outputs
    with autocast_ctx:
        x_a = x[:, self.anchor_pairs_a]
        x_b = x[:, self.anchor_pairs_b]
        rd = x_a - x_b
        p = rd / (T_soft + rd.abs())
        ts = torch.einsum("btp,pk->btk", p, self.bit_matrix.to(p.dtype))

        # Hard STE (gumbel=False, hard=True path)
        sel_soft = F.softmax(ts / T_sel, dim=-1)                    # [B, n_tables, 2^nap]
        # Forward path: gather weights at argmax(ts)
        index = ts.argmax(dim=-1)                                    # [B, n_tables]
        # weights: [n_tables, 2^nap, n_outputs]
        # Take weights[t, index[b, t], :]
        n_tables = self.n_lookup_tables
        # Use advanced indexing: weights[table_idx, index, :]
        # build table_idx shape [1, n_tables] -> broadcast to [B, n_tables]
        table_idx = torch.arange(n_tables, device=x.device).unsqueeze(0).expand(B, -1)
        out_hard = self.weights[table_idx, index, :]                 # [B, n_tables, n_outputs]
        # Soft path for backward (same einsum as before but on sel_soft alone, not combined sel)
        out_soft = torch.einsum("btk,tko->bto", sel_soft, self.weights)
        out_t = out_hard.detach() + out_soft - out_soft.detach()

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
    print(f"  {label:<25s} {dt:6.2f} ms (fwd+bwd)  peak={peak:7.1f} MB")
    return dt, peak


# Equivalence check (eager) before benchmarking
print("\n=== Equivalence check ===")
mod_a = make(CONFIGS[1])
torch.manual_seed(42); x_test = torch.randn(B_TOK, CONFIGS[1]["input_dim"], device=DEVICE, requires_grad=True)
out_orig = mod_a(x_test); g_orig = torch.autograd.grad(out_orig.sum(), [x_test, mod_a.weights, mod_a.log_soft_score_temp, mod_a.log_select_temp], retain_graph=False)
import types
mod_a.forward = types.MethodType(output_ste_forward, mod_a)
torch.manual_seed(42); x_test = torch.randn(B_TOK, CONFIGS[1]["input_dim"], device=DEVICE, requires_grad=True)
out_new = mod_a(x_test); g_new = torch.autograd.grad(out_new.sum(), [x_test, mod_a.weights, mod_a.log_soft_score_temp, mod_a.log_select_temp], retain_graph=False)
print(f"  out match    max|Δ| = {(out_orig - out_new).abs().max():.2e}")
for name, (a, b) in zip(["dx","dw","dT_soft","dT_sel"], zip(g_orig, g_new)):
    print(f"  grad {name:8s}  max|Δ| = {(a-b).abs().max():.2e}")


for cfg in CONFIGS:
    print(f"\n=== {cfg['name']}  in={cfg['input_dim']} out={cfg['n_outputs']} "
          f"nap={cfg['nap']} tph={cfg['tph']} H={cfg['n_heads']}  B={B_TOK} ===")
    # Original
    mod = make(cfg)
    mod.forward = torch.compile(mod.forward, dynamic=True)
    x = torch.randn(B_TOK, cfg["input_dim"], device=DEVICE, requires_grad=True)
    t_orig, m_orig = bench(mod, x, "original (compiled)")
    # New STE-on-output
    mod = make(cfg)
    mod.forward = torch.compile(types.MethodType(output_ste_forward, mod), dynamic=True)
    x = torch.randn(B_TOK, cfg["input_dim"], device=DEVICE, requires_grad=True)
    t_new, m_new = bench(mod, x, "output-STE (compiled)")
    print(f"  -> speed: {(t_orig - t_new)/t_orig*100:+.1f}%  memory: {(m_new - m_orig)/m_orig*100:+.1f}%")
