"""Benchmark inference wallclock for:
  exp750 LUT-LM with bf16 LUT body + bf16 head
  exp738 vanilla with full bf16 (backbone + head)

Same input shape, same n_iters, same GPU. Reports tokens/sec and ms/forward.
"""
import sys, os, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get("NANOCHAT_ROOT", "/home/starost/nanochat")
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
os.environ.setdefault("FORCE_BMM_WGRAD", "1")

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = torch.device("cuda:0")

# Inference workload: same as training eval shape
B_BENCH = 24
SEQ_BENCH = 512

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, "tokenizer"))
VOCAB_SIZE = tokenizer.get_vocab_size()

# Fixed input batch for both models (same indices)
torch.manual_seed(0)
INPUT_IDS = torch.randint(0, VOCAB_SIZE, (B_BENCH, SEQ_BENCH), device=DEVICE)
TARGETS = torch.randint(0, VOCAB_SIZE, (B_BENCH, SEQ_BENCH), device=DEVICE)


def time_forward(model, n_warmup=5, n_iter=20):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            out = model(INPUT_IDS)
            torch.cuda.synchronize()
        # Timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            out = model(INPUT_IDS)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms_per_iter = (t1 - t0) / n_iter * 1000
    return ms_per_iter


def report(name, ms):
    tokens = B_BENCH * SEQ_BENCH
    print(f"\n{name}")
    print(f"  ms/forward: {ms:.2f}")
    print(f"  tokens/sec: {tokens * 1000 / ms:,.0f} ({tokens} tokens / batch)")


# ===== Load vanilla exp738 ===================================================
print("=" * 60)
print("Loading exp738 vanilla MinimalGPT")
print("=" * 60)
EXP738_DIR = "/home/starost/spiky/nanochat_exps/exp738_vanilla_bf16_bs48_4k"
with open(os.path.join(EXP738_DIR, "config.json")) as f:
    cfg738 = json.load(f)


class RotaryEmbedding738(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)


def _rh(x):
    a, b = x.chunk(2, dim=-1); return torch.cat([-b, a], dim=-1)


def apply_rope_v(q, k, cos, sin):
    cos = cos.to(q.dtype)[None, None, :, :]
    sin = sin.to(q.dtype)[None, None, :, :]
    return q*cos + _rh(q)*sin, k*cos + _rh(k)*sin


class VanAttn(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q, k = apply_rope_v(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class VanBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
        self.attn = VanAttn(n_embd, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd, bias=False))
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class VanGPT(nn.Module):
    def __init__(self, V, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(V, n_embd)
        head_dim = n_embd // n_head
        self.rope = RotaryEmbedding738(head_dim, seq_len)
        self.blocks = nn.ModuleList([VanBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V, bias=False)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        h = self.ln_f(x).to(self.head.weight.dtype)
        return self.head(h)


vanilla = VanGPT(VOCAB_SIZE, cfg738["n_embd"], cfg738["n_head"], cfg738["depth"], cfg738["seq_len"]).to(DEVICE)
ckpt738 = torch.load(os.path.join(EXP738_DIR, "checkpoint.pt"), weights_only=False, map_location="cpu")
vanilla.load_state_dict(ckpt738)
# Cast all params to bf16 (full bf16 deployment)
for p in vanilla.parameters():
    p.data = p.data.to(torch.bfloat16)
print(f"vanilla params: {sum(p.numel() for p in vanilla.parameters()):,}, all bf16: {all(p.dtype == torch.bfloat16 for p in vanilla.parameters())}")


# ===== Load LUT-LM exp750 ====================================================
print("\n" + "=" * 60)
print("Loading exp750 LUT-LM")
print("=" * 60)
EXP750_DIR = "/home/starost/spiky/nanochat_exps/exp750_FastMHL_exp737v2_clip_all_4K"
with open(os.path.join(EXP750_DIR, "config.json")) as f:
    cfg = json.load(f)

CONTEXT_SIZE = cfg["context_size"]; E = cfg["embedding_dim"]; D = cfg["residual_dim"]
H = cfg["n_heads"]; d_qk = cfg["d_qk"]; d_v = cfg["d_v"]
N_LAYERS = cfg["num_layers"]; ROPE_BASE = cfg.get("rope_base", 10000.0)

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
    use_bf16=cfg.get("soft_use_bf16", True))


def make_lut(input_dim, n_heads, n_outputs, nap_key, tph_key, offset):
    return FastMultiHeadLUT(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=cfg[nap_key], tables_per_head=cfg[tph_key],
        random_seed=cfg["random_seed"] + offset, device=DEVICE, **_FAST_KWARGS)


class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps
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
    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B*T, E))
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D).float()
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid).to(self.unembedder.weight.dtype)
        return self.unembedder(x_resid)


lutlm = Model().to(DEVICE)
ckpt750 = torch.load(os.path.join(EXP750_DIR, "checkpoint.pt"), weights_only=False, map_location="cpu")
lutlm.load_state_dict(ckpt750["model_state_dict"])
# Cast the unembedder (only fp32 weight in deployment) to bf16
lutlm.unembedder.weight.data = lutlm.unembedder.weight.data.to(torch.bfloat16)
print(f"LUT-LM params: {sum(p.numel() for p in lutlm.parameters()):,}")
print(f"  LUT body dtype: {lutlm.layers[0].residual_lut.weights.dtype}")
print(f"  unembedder dtype: {lutlm.unembedder.weight.dtype}")


# ===== Benchmark =============================================================
print("\n" + "=" * 60)
print(f"Benchmark: B={B_BENCH}, T={SEQ_BENCH}, n_warmup=5, n_iter=20")
print("=" * 60)

ms_vanilla = time_forward(vanilla)
report("vanilla exp738 (full bf16)", ms_vanilla)

ms_lutlm = time_forward(lutlm)
report("exp750 LUT-LM (bf16 LUT + bf16 head)", ms_lutlm)

print("\n" + "=" * 60)
print(f"vanilla / LUT-LM speed ratio:  {ms_vanilla / ms_lutlm:.2f}x")
print(f"LUT-LM ms is {(ms_lutlm / ms_vanilla - 1)*100:+.1f}% vs vanilla")
print("=" * 60)
