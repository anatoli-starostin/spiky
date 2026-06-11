"""Diagnostic: compare fp32 vs bf16 unembedder forward+backward on LUT-LM.

Builds two identical Model instances (same seed) and runs one forward+backward
on the same batch. The only difference is whether self.unembedder is wrapped
in torch.amp.autocast(bf16). Reports loss values and gradient statistics for
the head, x_resid, and a sample residual_lut.
"""
import sys, os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

# Force bmm-wgrad (matches exp737 v2 / exp742 launch env)
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
E = cfg["embedding_dim"]
D = cfg["residual_dim"]
H = cfg["n_heads"]
d_qk = cfg["d_qk"]
d_v = cfg["d_v"]
N_LAYERS = cfg["num_layers"]
DEVICE_BS = cfg["device_batch_size"]

# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()

# Grab one batch (same for both runs)
loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split="train", device=DEVICE
)
x_batch, y_batch = next(loader)
print(f"Batch: x={x_batch.shape}, y={y_batch.shape}, V={VOCAB_SIZE}")


# --- LUT factories (minimal copy of exp737 v2 train.py) -----------------------
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


def make_qk(offset):
    return FastMultiHeadLUT(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg["qkv_input_nap"], tables_per_head=cfg["qkv_tph"],
        random_seed=cfg["random_seed"] + offset, device=DEVICE, **_FAST_KWARGS)


def make_v(offset):
    return FastMultiHeadLUT(input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg["v_input_nap"], tables_per_head=cfg["v_tph"],
        random_seed=cfg["random_seed"] + offset, device=DEVICE, **_FAST_KWARGS)


def make_out(offset):
    return FastMultiHeadLUT(input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg["out_input_nap"], tables_per_head=cfg["out_tph"],
        random_seed=cfg["random_seed"] + offset, device=DEVICE, **_FAST_KWARGS)


def make_residual(offset):
    return FastMultiHeadLUT(input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg["residual_input_nap"], tables_per_head=cfg["residual_tph"],
        random_seed=cfg["random_seed"] + offset, device=DEVICE, **_FAST_KWARGS)


def make_emb_resid(offset):
    return FastMultiHeadLUT(input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg["emb_resid_input_nap"], tables_per_head=cfg["emb_resid_tph"],
        random_seed=cfg["random_seed"] + offset, device=DEVICE, **_FAST_KWARGS)


class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        mean_abs = x.abs().mean(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x / mean_abs) * self.weight


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
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    q = (q * cos) + (rot(q) * sin)
    k = (k * cos) + (rot(k) * sin)
    return q, k


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_lut = make_qk(layer_idx)
        self.v_lut = make_v(200 + layer_idx)
        self.out_proj = make_out(400 + layer_idx)
        self.residual_lut = make_residual(600 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E)
        self.ln_resid = MeanAbsNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)
        x_pre = self.ln_pre(x_flat)
        qk_out = self.qk_lut(x_pre).float()
        q_vec = self.q_norm(qk_out[..., :d_qk])
        k_vec = self.k_norm(qk_out[..., d_qk:2 * d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v_vec = self.v_lut(x_pre).float()
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e = self.out_proj(out_in).squeeze(1).float()
        x_lut_next_flat = x_flat + out_e
        r_in = self.ln_resid(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D).float()
        return x_lut_next_flat.reshape(B, T, E), r_out


class Model(nn.Module):
    def __init__(self, head_dtype_mode):
        super().__init__()
        self.head_dtype_mode = head_dtype_mode  # 'fp32' or 'bf16'
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.emb_resid_lut = make_emb_resid(800)
        self.ln_emb_resid = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B * T, E))
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D).float()
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        # Capture x_resid for grad inspection
        x_resid.retain_grad()
        self._last_x_resid = x_resid
        if self.head_dtype_mode == 'fp32':
            logits = self.unembedder(x_resid)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = self.unembedder(x_resid)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1), ignore_index=-1)
        return logits, loss


def stats(t):
    if t is None:
        return "None"
    t = t.detach().float()
    return f"dtype={t.dtype} shape={tuple(t.shape)} mean={t.mean().item():+.4e} std={t.std().item():.4e} absmax={t.abs().max().item():.4e}"


def run(mode):
    torch.manual_seed(cfg['random_seed'])
    torch.cuda.manual_seed_all(cfg['random_seed'])
    model = Model(head_dtype_mode=mode).to(DEVICE)
    model.train()
    logits, loss = model(x_batch, targets=y_batch)
    loss.backward()
    return {
        "logits": logits,
        "loss": loss.detach().item(),
        "x_resid_grad": model._last_x_resid.grad,
        "unembed_w_grad": model.unembedder.weight.grad,
        "residual_lut_layer0_grad": model.layers[0].residual_lut.weights.grad,
        "residual_lut_layer3_grad": model.layers[3].residual_lut.weights.grad,
    }


print("\n=== Path A: fp32 unembedder ===")
A = run("fp32")
print(f"  loss              = {A['loss']:.6f}")
print(f"  logits            : {stats(A['logits'])}")
print(f"  grad x_resid      : {stats(A['x_resid_grad'])}")
print(f"  grad unembed.w    : {stats(A['unembed_w_grad'])}")
print(f"  grad resid_lut[0] : {stats(A['residual_lut_layer0_grad'])}")
print(f"  grad resid_lut[3] : {stats(A['residual_lut_layer3_grad'])}")

print("\n=== Path B: bf16 unembedder (autocast) ===")
B = run("bf16")
print(f"  loss              = {B['loss']:.6f}")
print(f"  logits            : {stats(B['logits'])}")
print(f"  grad x_resid      : {stats(B['x_resid_grad'])}")
print(f"  grad unembed.w    : {stats(B['unembed_w_grad'])}")
print(f"  grad resid_lut[0] : {stats(B['residual_lut_layer0_grad'])}")
print(f"  grad resid_lut[3] : {stats(B['residual_lut_layer3_grad'])}")

print("\n=== Diff B - A ===")
print(f"  loss delta              = {B['loss'] - A['loss']:+.6f}")
for k in ["x_resid_grad", "unembed_w_grad", "residual_lut_layer0_grad", "residual_lut_layer3_grad"]:
    a, b = A[k].detach().float(), B[k].detach().float()
    diff = (b - a).flatten()
    rel = diff.abs() / (a.abs().flatten() + 1e-12)
    cos = F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
    sign_flip = ((a > 0) != (b > 0)).float().mean().item()
    print(f"  {k:25s}  cos={cos:.6f}  rel_rms={diff.pow(2).mean().sqrt().item():.4e}  sign_flip_rate={sign_flip:.4f}")
