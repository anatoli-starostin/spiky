"""Diagnostic: track per-step drift between fp32-head and bf16-head LUT-LM
trajectories over 100 training steps, sharing identical micro-batches and seed.

Reports:
  - per-step loss values for both
  - residual_lut[0] grad stats (mean, std, max abs diff between heads)
  - sign-flip rate of (m*b1 + g*(1-b1)) between the two heads — proxy for
    "wrong Lion update direction"
  - Lion master per-step delta magnitude for both trajectories (how far the
    master moves each step) — chaotic vs systematic divergence
"""
import sys, os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get("NANOCHAT_ROOT", "/home/starost/nanochat")
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
os.environ.setdefault("FORCE_BMM_WGRAD", "1")

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = torch.device("cuda:0")
EXP_DIR = "/home/starost/spiky/nanochat_exps/exp737_FastMHL_bf16_weights_4K"
with open(os.path.join(EXP_DIR, "config.json")) as f:
    cfg = json.load(f)

CONTEXT_SIZE = cfg["context_size"]
E = cfg["embedding_dim"]; D = cfg["residual_dim"]
H = cfg["n_heads"]; d_qk = cfg["d_qk"]; d_v = cfg["d_v"]
N_LAYERS = cfg["num_layers"]
DEVICE_BS = cfg["device_batch_size"]
TOTAL_BS = cfg["total_batch_size"]
LR_WARMUP = cfg["lr_warmup_fraction"]
ROPE_BASE = cfg.get("rope_base", 10000.0)

BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()

_WEIGHT_DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg.get("weight_dtype", "fp32")]
_FAST_KWARGS = dict(
    forward_mode=cfg.get("forward_mode", "hard"),
    backward_mode=cfg.get("backward_mode", "ball"),
    weight_dtype=_WEIGHT_DTYPE,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get("mhlut_init_std", 0.001),
    soft_score_temp=cfg.get("soft_score_temp", 0.5),
    select_temp=cfg.get("select_temp", 0.5),
    learnable_temps=cfg.get("soft_learnable_temps", True),
    use_bf16=cfg.get("soft_use_bf16", True),
)


def make_lut(input_dim, n_heads, n_outputs, nap_key, tph_key, offset):
    return FastMultiHeadLUT(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=cfg[nap_key], tables_per_head=cfg[tph_key],
        random_seed=cfg["random_seed"] + offset, device=DEVICE, **_FAST_KWARGS)


class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return (x / x.abs().mean(dim=-1, keepdim=True).clamp_min(self.eps)) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos().to(DEVICE), persistent=False)
        self.register_buffer("sin", emb.sin().to(DEVICE), persistent=False)


def apply_rope(q, k, cos, sin):
    def rot(x):
        x1, x2 = x.chunk(2, dim=-1); return torch.cat([-x2, x1], dim=-1)
    return (q*cos) + (rot(q)*sin), (k*cos) + (rot(k)*sin)


class LUTBlock(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.qk_lut = make_lut(E, H, 2*d_qk, "qkv_input_nap", "qkv_tph", i)
        self.v_lut = make_lut(E, H, d_v, "v_input_nap", "v_tph", 200+i)
        self.out_proj = make_lut(H*d_v, 1, E, "out_input_nap", "out_tph", 400+i)
        self.residual_lut = make_lut(E, 1, D, "residual_input_nap", "residual_tph", 600+i)
        self.q_norm = nn.LayerNorm(d_qk); self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E); self.ln_resid = MeanAbsNorm(E)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B*T, E)
        x_pre = self.ln_pre(x_flat)
        qk_out = self.qk_lut(x_pre).float()
        q_vec = self.q_norm(qk_out[..., :d_qk]); k_vec = self.k_norm(qk_out[..., d_qk:2*d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v = self.v_lut(x_pre).float().reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
        out_e = self.out_proj(out_in).squeeze(1).float()
        x_lut_next_flat = x_flat + out_e
        r_in = self.ln_resid(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D).float()
        return x_lut_next_flat.reshape(B, T, E), r_out


class Model(nn.Module):
    def __init__(self, head_dtype_mode):
        super().__init__()
        self.head_dtype_mode = head_dtype_mode
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.emb_resid_lut = make_lut(E, 1, D, "emb_resid_input_nap", "emb_resid_tph", 800)
        self.ln_emb_resid = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, CONTEXT_SIZE, base=ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)
    def forward(self, tokens, targets):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B*T, E))
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D).float()
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        if self.head_dtype_mode == "fp32":
            logits = self.unembedder(x_resid)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.unembedder(x_resid)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1), ignore_index=-1)
        return loss


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp["lr"], grp["betas"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None: continue
                st = self.state[p]; is_low = p.dtype != torch.float32
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    if is_low: st["master"] = p.detach().to(torch.float32).clone()
                m = st["exp_avg"]
                g_f = p.grad.to(torch.float32) if p.grad.dtype != torch.float32 else p.grad
                if is_low:
                    master = st["master"]
                    if wd != 0: master.mul_(1.0 - lr*wd)
                    upd = (m*b1 + g_f*(1.0-b1)).sign_()
                    master.add_(upd, alpha=-lr); m.mul_(b2).add_(g_f, alpha=1.0-b2)
                    p.data.copy_(master)
                else:
                    if wd != 0: p.mul_(1.0 - lr*wd)
                    upd = (m*b1 + g_f*(1.0-b1)).sign_()
                    p.add_(upd, alpha=-lr); m.mul_(b2).add_(g_f, alpha=1.0-b2)


def setup(mode):
    torch.manual_seed(cfg["random_seed"])
    torch.cuda.manual_seed_all(cfg["random_seed"])
    m = Model(head_dtype_mode=mode).to(DEVICE)
    lut_params, decay_params, tok_emb_params, nodecay_params = [], [], [], []
    for name, p in m.named_parameters():
        if not p.requires_grad: continue
        if p.ndim >= 3: lut_params.append(p)
        elif name.startswith("tok_emb_E."): tok_emb_params.append(p)
        elif p.ndim == 2: decay_params.append(p)
        else: nodecay_params.append(p)
    adam = torch.optim.AdamW([
        dict(params=decay_params, lr=cfg["adam_lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg["weight_decay"]),
        dict(params=tok_emb_params + nodecay_params, lr=cfg["adam_lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    ])
    lion = Lion([dict(params=lut_params, lr=cfg["lut_lr"], weight_decay=0.0)])
    return m, adam, lion, lut_params


# Load a batch
loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split="train", device=DEVICE)
batches = [next(loader) for _ in range(100)]


def lr_scale(step, total, warm_frac):
    w = int(round(warm_frac * total))
    if step <= w: return step / max(w, 1)
    p = (step - w) / max(total - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def run(mode, n_steps=100, snapshot_param_idx=0):
    """snapshot_param_idx: which LUT param to track per-step master."""
    m, adam, lion, lut_params = setup(mode)
    tracked = lut_params[snapshot_param_idx]
    initial_master = tracked.detach().to(torch.float32).clone()
    losses = []
    grad_history = []  # snapshot grad each step
    master_traj = [initial_master.clone()]  # the master after each Lion step
    for step in range(1, n_steps + 1):
        scale = lr_scale(step, 4000, LR_WARMUP)
        for o in [adam, lion]:
            for g in o.param_groups:
                g.setdefault("initial_lr", g["lr"])
                g["lr"] = g["initial_lr"] * scale
        for o in [adam, lion]: o.zero_grad()
        x, y = batches[(step - 1) % len(batches)]
        loss = m(x, y)
        loss.backward()
        losses.append(loss.item())
        # snapshot grad on tracked param BEFORE step
        grad_history.append(tracked.grad.detach().to(torch.float32).clone())
        for o in [adam, lion]: o.step()
        # snapshot master AFTER step
        st = lion.state[tracked]
        master_traj.append(st["master"].detach().clone() if "master" in st
                           else tracked.detach().to(torch.float32).clone())
    return losses, grad_history, master_traj


print("=== Running fp32-head trajectory (100 steps) ===")
loss_A, grad_A, master_A = run("fp32", n_steps=100)
print("=== Running bf16-head trajectory (100 steps) ===")
loss_B, grad_B, master_B = run("bf16", n_steps=100)

print(f"\n=== Per-step loss A vs B ===")
print(f"{'step':>5s} {'loss_A':>9s} {'loss_B':>9s} {'B-A':>10s}")
for s in [1, 5, 10, 25, 50, 75, 100]:
    da = loss_A[s-1]; db = loss_B[s-1]
    print(f"{s:>5d} {da:9.5f} {db:9.5f} {db-da:+10.5f}")

print(f"\n=== residual_lut[0] grad stats per step ===")
print(f"{'step':>5s} {'std_A':>10s} {'std_B':>10s} {'cos(A,B)':>10s} {'sign_flip%':>12s}")
for s in [1, 5, 10, 25, 50, 75, 100]:
    gA = grad_A[s-1].flatten(); gB = grad_B[s-1].flatten()
    cos = F.cosine_similarity(gA.unsqueeze(0), gB.unsqueeze(0)).item()
    flip = ((gA > 0) != (gB > 0)).float().mean().item() * 100
    print(f"{s:>5d} {gA.std().item():10.4e} {gB.std().item():10.4e} {cos:10.6f} {flip:12.4f}")

print(f"\n=== Master delta between A and B (cumulative) ===")
print(f"{'step':>5s} {'||master_A-master_B||':>23s} {'||master_A-init||':>20s} {'||master_B-init||':>20s}")
init = master_A[0].flatten()
for s in [1, 5, 10, 25, 50, 75, 100]:
    diff = (master_A[s] - master_B[s]).flatten().norm().item()
    da = (master_A[s].flatten() - init).norm().item()
    db = (master_B[s].flatten() - init).norm().item()
    print(f"{s:>5d} {diff:23.6e} {da:20.6e} {db:20.6e}")
