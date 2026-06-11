"""Load exp750 checkpoint and evaluate val_bpb with:
  (1) the original storage: bf16 LUT body + fp32 head (baseline)
  (2) bf16 LUT body + bf16 quantized head (deployment scenario)

Tests whether the bf16-head training-time drift we diagnosed (chaotic
divergence from sign-flips during Lion + sparse LUT updates) is purely a
training-time issue. At inference there's no optimizer step, so bf16 head
should be ~free if the precision-induced logit error is small.
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
from nanochat.loss_eval import evaluate_bpb
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = torch.device("cuda:0")
EXP_DIR = "/home/starost/spiky/nanochat_exps/exp750_FastMHL_exp737v2_clip_all_4K"
with open(os.path.join(EXP_DIR, "config.json")) as f:
    cfg = json.load(f)

CONTEXT_SIZE = cfg["context_size"]
E = cfg["embedding_dim"]; D = cfg["residual_dim"]
H = cfg["n_heads"]; d_qk = cfg["d_qk"]; d_v = cfg["d_v"]
N_LAYERS = cfg["num_layers"]
DEVICE_BS = cfg["device_batch_size"]
EVAL_STEPS = cfg["eval_steps"]
ROPE_BASE = cfg.get("rope_base", 10000.0)

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, "tokenizer"))
VOCAB_SIZE = tokenizer.get_vocab_size()
token_bytes = get_token_bytes(device=DEVICE)

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
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


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
        self.emb_resid_lut = make_lut(E, 1, D, "emb_resid_input_nap", "emb_resid_tph", 800)
        self.ln_emb_resid = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, CONTEXT_SIZE, base=ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)
    def get_device(self): return self.tok_emb_E.weight.device
    def forward(self, tokens, targets=None, loss_reduction="mean"):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B*T, E))
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D).float()
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        # Cast to match unembedder's storage dtype (handles bf16 head case)
        x_resid = x_resid.to(self.unembedder.weight.dtype)
        logits = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1)
        return logits


model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, "checkpoint.pt"), weights_only=False, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded exp750 checkpoint (step {ckpt['step']}, best_bpb {ckpt['best_val_bpb']:.4f})")
print(f"  LUT body dtype:    {model.layers[0].residual_lut.weights.dtype}")
print(f"  unembedder dtype:  {model.unembedder.weight.dtype}")


def make_val_loader():
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BS, CONTEXT_SIZE, split="val", device=DEVICE)


# Baseline: original storage (bf16 LUT + fp32 head) ---------------------------
model.eval()
with torch.no_grad():
    val_loader = make_val_loader()
    bpb_orig = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
print(f"\n[bf16 LUT + fp32 head, original]       val_bpb = {bpb_orig:.4f}")


# Cast unembedder to bf16 -----------------------------------------------------
model.unembedder.weight.data = model.unembedder.weight.data.to(torch.bfloat16)
print(f"\nUnembedder cast to bf16: dtype={model.unembedder.weight.dtype}")
with torch.no_grad():
    val_loader = make_val_loader()
    bpb_bf16_head = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
print(f"[bf16 LUT + bf16 head]                 val_bpb = {bpb_bf16_head:.4f}")
print(f"Delta vs fp32 head: {(bpb_bf16_head - bpb_orig)*1000:+.2f} mb")
