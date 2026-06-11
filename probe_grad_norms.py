"""Measure gradient norms per parameter group in exp737 v2 at init and after
~100 steps of training (warm-up phase) so we know what magnitude
clip_grad_norm_ would actually act on.

Forks the exp737 v2 train.py model + optimizer setup but trims it to just
N_STEPS_PROBE steps with verbose grad-norm logging.
"""
import sys, os, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
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

torch.manual_seed(cfg['random_seed'])
torch.cuda.manual_seed_all(cfg['random_seed'])

CONTEXT_SIZE = cfg["context_size"]
E = cfg["embedding_dim"]; D = cfg["residual_dim"]
H = cfg["n_heads"]; d_qk = cfg["d_qk"]; d_v = cfg["d_v"]
N_LAYERS = cfg["num_layers"]
DEVICE_BS = cfg["device_batch_size"]
TOTAL_BS = cfg["total_batch_size"]
LR_WARMUP = cfg["lr_warmup_fraction"]

BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()

loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split="train", device=DEVICE
)

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
        random_seed=cfg["random_seed"] + offset, device=DEVICE, **_FAST_KWARGS,
    )


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
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.emb_resid_lut = make_lut(E, 1, D, "emb_resid_input_nap", "emb_resid_tph", 800)
        self.ln_emb_resid = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, CONTEXT_SIZE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)
    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B*T, E))
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D).float()
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits = self.unembedder(x_resid)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


model = Model().to(DEVICE)

lut_params, tok_emb_params, decay_params, nodecay_params = [], [], [], []
for name, p in model.named_parameters():
    if not p.requires_grad: continue
    if p.ndim >= 3: lut_params.append(p)
    elif name.startswith("tok_emb_E."): tok_emb_params.append(p)
    elif p.ndim == 2: decay_params.append(p)
    else: nodecay_params.append(p)


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp["lr"], grp["betas"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None: continue
                st = self.state[p]
                is_low = p.dtype != torch.float32
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


adam = torch.optim.AdamW([
    dict(params=decay_params, lr=cfg["adam_lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg["weight_decay"]),
    dict(params=tok_emb_params + nodecay_params, lr=cfg["adam_lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
])
lion = Lion([dict(params=lut_params, lr=cfg["lut_lr"], weight_decay=0.0)])

opts = [adam, lion]
for o in opts:
    for g in o.param_groups:
        g["initial_lr"] = g["lr"]


def total_norm(params):
    sq = 0.0
    for p in params:
        if p.grad is None: continue
        sq += (p.grad.detach().float() ** 2).sum().item()
    return math.sqrt(sq)


N_STEPS_PROBE = 100
grad_accum = max(1, TOTAL_BS // (DEVICE_BS * CONTEXT_SIZE))


def lr_scale(step, total, warm_frac):
    w = int(round(warm_frac * total))
    if step <= w: return step / max(w, 1)
    p = (step - w) / max(total - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


print(f"Probing {N_STEPS_PROBE} steps at {DEVICE_BS} dbs x {grad_accum} accum (exp737 v2 schedule clamp).")
print(f"{'step':>5s} | {'lut':>10s} | {'decay(unembed)':>14s} | {'tok_emb':>10s} | {'nodecay':>10s} | {'GLOBAL_ALL':>10s}")
for step in range(1, N_STEPS_PROBE + 1):
    scale = lr_scale(step, 4000, LR_WARMUP)
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["initial_lr"] * scale
    for o in opts:
        o.zero_grad()
    for _ in range(grad_accum):
        x, y = next(loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()

    if step in (1, 2, 5, 10, 25, 50, 75, 100):
        n_lut = total_norm(lut_params)
        n_decay = total_norm(decay_params)
        n_tok = total_norm(tok_emb_params)
        n_nodecay = total_norm(nodecay_params)
        n_all = total_norm(lut_params + decay_params + tok_emb_params + nodecay_params)
        print(f"{step:>5d} | {n_lut:10.2f} | {n_decay:14.4f} | {n_tok:10.4f} | {n_nodecay:10.4f} | {n_all:10.2f}")

    for o in opts:
        o.step()
