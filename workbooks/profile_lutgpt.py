"""
Profile LUTGPT (Part 5 of workbooks/nanochat_walkthrough.py) on the current VM.

Usage:
    # Notebook's setting — torch.compile disabled (eager fallback)
    python workbooks/profile_lutgpt.py --dynamo disable

    # With torch.compile enabled — LUT module's @torch.compile decorators fire
    python workbooks/profile_lutgpt.py --dynamo enable

Optional flags: --n-warmup, --n-measure, --trace-out (chrome trace JSON).
Env var: NANOCHAT_PATH (default /home/starost/nanochat) — where the nanochat
package lives so we can reuse its tokenizer and dataloader.

What it does:
  1. Builds the exact LUTGPT model from the notebook (dims match exp666).
  2. Pulls real batches from nanochat's tokenizing_distributed_data_loader_bos_bestfit.
  3. Runs warmup + measurement steps and reports wall-clock per step.
  4. Times fwd / bwd / opt separately.
  5. Captures a torch.profiler trace and prints top ops by self CUDA time.
"""

import os, sys, time, argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Toggle dynamo BEFORE importing the LUT module (decorators bind at import).
parser = argparse.ArgumentParser()
parser.add_argument("--dynamo", choices=["disable", "enable"], required=True)
parser.add_argument("--n-warmup", type=int, default=10)
parser.add_argument("--n-measure", type=int, default=30)
parser.add_argument("--profile-step", type=int, default=1, help="how many steps to capture in the trace")
parser.add_argument("--trace-out", type=str, default=None, help="path to write chrome trace json")
args = parser.parse_args()

import torch._dynamo
if args.dynamo == "disable":
    torch._dynamo.config.disable = True
    print("[cfg] torch._dynamo.config.disable = True (eager only)")
else:
    print("[cfg] torch.compile enabled")

NANOCHAT_PATH = os.environ.get("NANOCHAT_PATH", "/home/starost/nanochat")
sys.path.insert(0, NANOCHAT_PATH)

from nanochat.common import COMPUTE_DTYPE, get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = "cuda"
BASE_DIR = get_base_dir()
print(f"[cfg] device={DEVICE}  dtype={COMPUTE_DTYPE}  base_dir={BASE_DIR}")

# -- Tokenizer / dataloader ---------------------------------------------------
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)

SEQ_LEN = 512
DEVICE_BATCH_SIZE = 8
train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, split="train", device=DEVICE
)

# -- LUTGPT (verbatim from notebook Part 5) -----------------------------------
LUT_E, LUT_H, LUT_D_QK, LUT_D_V, LUT_N_LAYERS = 384, 6, 64, 64, 6
LUT_QK_NAP, LUT_QK_TPH, LUT_QK_S = 4, 64, 4
LUT_V_NAP,  LUT_V_TPH,  LUT_V_S  = 6, 128, 4
LUT_OUT_NAP, LUT_OUT_TPH, LUT_OUT_NH = 6, 256, 6
LUT_RO_NAP,  LUT_RO_TPH,  LUT_RO_NH  = 6, 256, 6
LUT_ROPE_BASE, LUT_INIT_STD, LUT_SEED = 10000.0, 0.001, 42
LUT_ADAM_LR, LUT_LUT_LR, LUT_LUT_BETAS, LUT_WEIGHT_DECAY = 3e-4, 2e-4, (0.9, 0.95), 0.1

_LUT_HS_KWARGS = dict(
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=LUT_INIT_STD,
    backward_mode="hybrid_smooth",
    soft_score_temp=0.5,
    select_temp=0.5,
    learnable_temps=True,
    use_bf16=True,
    argmax_noise_eps=0.0,
)

def _make_lut(input_dim, n_heads, n_outputs, tables_per_head, n_anchor_pairs, seed):
    return TinyMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs, tables_per_head=tables_per_head,
        random_seed=seed, device=torch.device(DEVICE), **_LUT_HS_KWARGS,
    )

class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp["lr"], grp["betas"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None: continue
                g = p.grad
                st = self.state[p]
                if "exp_avg" not in st: st["exp_avg"] = torch.zeros_like(p)
                m = st["exp_avg"]
                if wd != 0: p.mul_(1.0 - lr * wd)
                update = (m * b1 + g * (1.0 - b1)).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1.0 - b2)

class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        Hqk = LUT_H * LUT_QK_S
        Hv  = LUT_H * LUT_V_S
        self.qk_lut   = _make_lut(LUT_E, Hqk, (2 * LUT_D_QK) // LUT_QK_S,
                                  LUT_QK_TPH, LUT_QK_NAP, LUT_SEED + layer_idx)
        self.v_lut    = _make_lut(LUT_E, Hv,  LUT_D_V // LUT_V_S,
                                  LUT_V_TPH,  LUT_V_NAP,  LUT_SEED + 200 + layer_idx)
        self.out_proj = _make_lut(LUT_H * LUT_D_V, LUT_OUT_NH, LUT_E // LUT_OUT_NH,
                                  LUT_OUT_TPH, LUT_OUT_NAP, LUT_SEED + 400 + layer_idx)
        self.q_norm = nn.LayerNorm(LUT_D_QK)
        self.k_norm = nn.LayerNorm(LUT_D_QK)
        self.ln_pre = MeanAbsNorm(LUT_E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, LUT_E)
        x_pre  = self.ln_pre(x_flat)
        qk_out = self.qk_lut(x_pre)
        qk_per_head = qk_out.reshape(B * T, LUT_H, 2 * LUT_D_QK)
        q_vec = self.q_norm(qk_per_head[..., :LUT_D_QK])
        k_vec = self.k_norm(qk_per_head[..., LUT_D_QK:])
        q = q_vec.reshape(B, T, LUT_H, LUT_D_QK).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, LUT_H, LUT_D_QK).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v_out = self.v_lut(x_pre)
        v_vec = v_out.reshape(B * T, LUT_H, LUT_D_V)
        v = v_vec.reshape(B, T, LUT_H, LUT_D_V).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, LUT_H * LUT_D_V)
        out_e = self.out_proj(out_in).reshape(B * T, LUT_E)
        return (x_flat + out_e).reshape(B, T, LUT_E)

class LUTGPT(nn.Module):
    def __init__(self, vocab_size, seq_len):
        super().__init__()
        torch.manual_seed(LUT_SEED)
        self.tok_emb_E = nn.Embedding(vocab_size, LUT_E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.unembedder = nn.Linear(LUT_E, vocab_size, bias=False)
        self.rope = RotaryEmbedding(LUT_D_QK, max_seq_len=seq_len, base=LUT_ROPE_BASE, device=torch.device(DEVICE))
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(LUT_N_LAYERS)])
        self.read_out = _make_lut(LUT_E, LUT_RO_NH, LUT_E // LUT_RO_NH, LUT_RO_TPH, LUT_RO_NAP, LUT_SEED + 800)
        self.ln_final = nn.LayerNorm(LUT_E)

    def setup_optimizer(self, adam_lr=LUT_ADAM_LR, lut_lr=LUT_LUT_LR, lut_betas=LUT_LUT_BETAS, weight_decay=LUT_WEIGHT_DECAY):
        lut_params, tok_emb_params, decay_params, nodecay_params = [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if p.ndim >= 3: lut_params.append(p)
            elif name.startswith("tok_emb_E."): tok_emb_params.append(p)
            elif p.ndim == 2: decay_params.append(p)
            else: nodecay_params.append(p)
        adam_groups = [
            dict(params=decay_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
            dict(params=tok_emb_params + nodecay_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        ]
        adam_opt = torch.optim.AdamW(adam_groups)
        lion_opt = Lion([dict(params=lut_params, lr=lut_lr, weight_decay=0.0)], lr=lut_lr, betas=lut_betas)
        return [adam_opt, lion_opt]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb_E(idx)
        for layer in self.layers:
            x = layer(x, self.rope.cos, self.rope.sin)
        ro = self.read_out(x.reshape(B * T, LUT_E)).reshape(B, T, LUT_E)
        x = x + ro
        logits = self.unembedder(self.ln_final(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction="mean", ignore_index=-1)
        return logits

print("[build] constructing LUTGPT...")
model = LUTGPT(vocab_size=tokenizer.get_vocab_size(), seq_len=SEQ_LEN).to(DEVICE)
model.train()
n_params = sum(p.numel() for p in model.parameters())
print(f"[build] params={n_params:,}")

opts = model.setup_optimizer()

# -- Warmup -------------------------------------------------------------------
def step_once():
    x, y = next(train_loader)
    for opt in opts: opt.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    for opt in opts: opt.step()
    return loss

print(f"[warm] running {args.n_warmup} warmup steps (compile, allocator settle)...")
torch.cuda.synchronize()
t_warm = time.time()
for i in range(args.n_warmup):
    t0 = time.time()
    loss = step_once()
    torch.cuda.synchronize()
    print(f"  warm step {i:02d}  {(time.time()-t0)*1000:7.1f} ms  loss={loss.item():.3f}")
print(f"[warm] total {(time.time()-t_warm):.1f}s")

# -- Measure ------------------------------------------------------------------
torch.cuda.synchronize()
t_meas = time.time()
times_ms = []
for i in range(args.n_measure):
    t0 = time.time()
    step_once()
    torch.cuda.synchronize()
    times_ms.append((time.time() - t0) * 1000)
mean_ms = sum(times_ms) / len(times_ms)
sorted_t = sorted(times_ms)
p50 = sorted_t[len(sorted_t)//2]
p95 = sorted_t[int(len(sorted_t)*0.95)]
print(f"\n[meas] over {args.n_measure} steps:  mean={mean_ms:.1f}ms  median={p50:.1f}ms  p95={p95:.1f}ms  min={min(times_ms):.1f}ms  max={max(times_ms):.1f}ms")

# -- Per-phase timing (fwd / bwd / opt) ---------------------------------------
def phase_timing(n=10):
    fwd, bwd, opt_t = [], [], []
    for _ in range(n):
        x, y = next(train_loader)
        for o in opts: o.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t = time.time()
        loss = model(x, y)
        torch.cuda.synchronize(); fwd.append((time.time()-t)*1000)
        t = time.time()
        loss.backward()
        torch.cuda.synchronize(); bwd.append((time.time()-t)*1000)
        t = time.time()
        for o in opts: o.step()
        torch.cuda.synchronize(); opt_t.append((time.time()-t)*1000)
    return fwd, bwd, opt_t

fwd, bwd, opt_t = phase_timing(10)
print(f"[phase] fwd  mean={sum(fwd)/len(fwd):.1f}ms  median={sorted(fwd)[len(fwd)//2]:.1f}ms")
print(f"[phase] bwd  mean={sum(bwd)/len(bwd):.1f}ms  median={sorted(bwd)[len(bwd)//2]:.1f}ms")
print(f"[phase] opt  mean={sum(opt_t)/len(opt_t):.1f}ms  median={sorted(opt_t)[len(opt_t)//2]:.1f}ms")

# -- torch.profiler trace ------------------------------------------------------
from torch.profiler import profile, ProfilerActivity, record_function

print(f"\n[prof] capturing {args.profile_step}-step profiler trace...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False, with_stack=False, profile_memory=False,
) as prof:
    for _ in range(args.profile_step):
        x, y = next(train_loader)
        with record_function("zero_grad"):
            for o in opts: o.zero_grad(set_to_none=True)
        with record_function("forward"):
            loss = model(x, y)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            for o in opts: o.step()
        torch.cuda.synchronize()

print("\n[prof] top 25 by self CUDA time:")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

if args.trace_out:
    prof.export_chrome_trace(args.trace_out)
    print(f"[prof] wrote chrome trace -> {args.trace_out}")

# -- Phase totals from profiler ------------------------------------------------
print("\n[prof] phase totals (cuda):")
for evt in prof.key_averages():
    if evt.key in ("forward", "backward", "optimizer", "zero_grad"):
        # torch 2.11 renamed cuda_time_total -> device_time_total
        cuda_total = getattr(evt, "device_time_total", None) or getattr(evt, "cuda_time_total", 0)
        print(f"  {evt.key:12s}  self_cpu={evt.self_cpu_time_total/1000:8.1f}ms  cuda_total={cuda_total/1000:8.1f}ms")
print("\nDONE")
