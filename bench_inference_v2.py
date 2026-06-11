"""Comprehensive inference benchmark:
  exp750 LUT-LM bf16 (LUT + head bf16, SDPA fp32)         — baseline
  exp750 LUT-LM bf16 + bf16 SDPA                           — q,k,v cast to bf16 just before SDPA
  exp738 vanilla full bf16                                 — reference

Reports:
  - val_bpb for each variant (correctness check)
  - ms/forward at B=24, T=512
  - tokens/sec
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
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = torch.device("cuda:0")
B_BENCH = 24
SEQ_BENCH = 512
EVAL_STEPS = 10

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, "tokenizer"))
VOCAB_SIZE = tokenizer.get_vocab_size()
token_bytes = get_token_bytes(device=DEVICE)


def time_forward(model, n_warmup=5, n_iter=20):
    model.eval()
    torch.manual_seed(0)
    INPUT_IDS = torch.randint(0, VOCAB_SIZE, (B_BENCH, SEQ_BENCH), device=DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(INPUT_IDS)
            torch.cuda.synchronize()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(INPUT_IDS)
        torch.cuda.synchronize(); t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000


def make_val_loader():
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B_BENCH, SEQ_BENCH, split="val", device=DEVICE)


# ===== Common helpers ========================================================
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


# ===== LUT-LM (exp750) =======================================================
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


class LUTBlock(nn.Module):
    def __init__(self, i, bf16_sdpa=False):
        super().__init__()
        self.bf16_sdpa = bf16_sdpa
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
        # ---- SDPA dtype switch ----
        if self.bf16_sdpa:
            q = q.to(torch.bfloat16); k = k.to(torch.bfloat16); v = v.to(torch.bfloat16)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.bf16_sdpa:
            attn = attn.float()
        out_in = attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
        out_e = self.out_proj(out_in).squeeze(1).float()
        x_lut_next_flat = x_flat + out_e
        r_in = self.ln_resid(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D).float()
        return x_lut_next_flat.reshape(B, T, E), r_out


class Model(nn.Module):
    def __init__(self, bf16_sdpa=False):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.emb_resid_lut = make_lut(E, 1, D, "emb_resid_input_nap", "emb_resid_tph", 800)
        self.ln_emb_resid = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, CONTEXT_SIZE, base=ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i, bf16_sdpa=bf16_sdpa) for i in range(N_LAYERS)])
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
        x_resid = self.ln_final(x_resid).to(self.unembedder.weight.dtype)
        logits = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), reduction=loss_reduction, ignore_index=-1)
        return logits


def setup_lutlm(bf16_sdpa):
    m = Model(bf16_sdpa=bf16_sdpa).to(DEVICE)
    ckpt = torch.load(os.path.join(EXP750_DIR, "checkpoint.pt"), weights_only=False, map_location="cpu")
    m.load_state_dict(ckpt["model_state_dict"])
    # cast unembedder to bf16 (deployment scenario)
    m.unembedder.weight.data = m.unembedder.weight.data.to(torch.bfloat16)
    return m


# ===== Vanilla (exp738) ======================================================
EXP738_DIR = "/home/starost/spiky/nanochat_exps/exp738_vanilla_bf16_bs48_4k"
with open(os.path.join(EXP738_DIR, "config.json")) as f:
    cfg738 = json.load(f)


def _rh(x):
    a, b = x.chunk(2, dim=-1); return torch.cat([-b, a], dim=-1)


def apply_rope_v(q, k, cos, sin):
    cos = cos.to(q.dtype)[None, None, :, :]; sin = sin.to(q.dtype)[None, None, :, :]
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


class VanRotary(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)


class VanGPT(nn.Module):
    def __init__(self, V, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(V, n_embd)
        self.rope = VanRotary(n_embd // n_head, seq_len)
        self.blocks = nn.ModuleList([VanBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V, bias=False)
    def get_device(self): return self.tok_emb.weight.device
    def forward(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.size()
        x = self.tok_emb(idx)
        for b in self.blocks: x = b(x, self.rope.cos, self.rope.sin)
        h = self.ln_f(x).to(self.head.weight.dtype)
        logits = self.head(h)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), reduction=loss_reduction, ignore_index=-1)
        return logits


def setup_vanilla():
    m = VanGPT(VOCAB_SIZE, cfg738["n_embd"], cfg738["n_head"], cfg738["depth"], cfg738["seq_len"]).to(DEVICE)
    ckpt = torch.load(os.path.join(EXP738_DIR, "checkpoint.pt"), weights_only=False, map_location="cpu")
    m.load_state_dict(ckpt)
    for p in m.parameters(): p.data = p.data.to(torch.bfloat16)
    return m


# ===== Run all four configs ==================================================
def run(name, model):
    model.eval()
    with torch.no_grad():
        loader = make_val_loader()
        bpb = evaluate_bpb(model, loader, EVAL_STEPS, token_bytes)
    ms = time_forward(model)
    tokens = B_BENCH * SEQ_BENCH
    print(f"\n{name}")
    print(f"  val_bpb   : {bpb:.4f}")
    print(f"  ms/forward: {ms:.2f}")
    print(f"  tokens/sec: {tokens * 1000 / ms:,.0f}")
    return bpb, ms


print("=" * 70)
print("LUT-LM exp750 with fp32 SDPA (baseline)")
print("=" * 70)
m = setup_lutlm(bf16_sdpa=False)
bpb_fp32_sdpa, ms_fp32_sdpa = run("exp750 (bf16 LUT + bf16 head, FP32 SDPA)", m)
del m; torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("LUT-LM exp750 with BF16 SDPA")
print("=" * 70)
m = setup_lutlm(bf16_sdpa=True)
bpb_bf16_sdpa, ms_bf16_sdpa = run("exp750 (bf16 LUT + bf16 head, BF16 SDPA)", m)
del m; torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("Vanilla exp738 (full bf16)")
print("=" * 70)
m = setup_vanilla()
bpb_van, ms_van = run("exp738 vanilla (full bf16)", m)
del m; torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"                                    val_bpb    ms/fwd     vs vanilla")
print(f"exp750 fp32 SDPA                    {bpb_fp32_sdpa:.4f}    {ms_fp32_sdpa:7.2f}   {ms_fp32_sdpa/ms_van:.1f}x slower")
print(f"exp750 bf16 SDPA                    {bpb_bf16_sdpa:.4f}    {ms_bf16_sdpa:7.2f}   {ms_bf16_sdpa/ms_van:.1f}x slower")
print(f"vanilla exp738 full bf16            {bpb_van:.4f}    {ms_van:7.2f}   1.0x")
